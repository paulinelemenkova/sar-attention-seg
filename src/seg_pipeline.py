import argparse
import glob
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

ND = -32768
TAU = -200
SEED = 42
DYN = ["VV", "VH", "RC", "TLOC"]


def read_dates(chips):
    d = pd.read_csv(os.path.join(chips, "chips_2025_dates.csv"))
    d["file"] = d["file"].astype(str).str.replace(r"\.tif$", "", regex=True)
    return d


def windows(chips, dates):
    out = []
    for f in sorted(glob.glob(os.path.join(chips, "chips_2025_b*_*.tif"))):
        name = os.path.basename(f)[:-4]
        rows = dates[dates.file == name].reset_index(drop=True)
        _, _, blk, split = name.split("_")
        out.append({"name": name, "path": f, "block": blk, "split": split, "rows": rows})
    return out


def load(w):
    with rasterio.open(w["path"]) as d:
        a = d.read()
        tr = d.transform
    n = len(w["rows"])
    if a.shape[0] != 5 * n + 2:
        sys.exit(f"{w['name']}: {a.shape[0]} bands, expected {5 * n + 2} for {n} dates")
    per = a[: 5 * n].reshape(n, 5, *a.shape[1:])
    dyn = per[:, :4].astype(np.int16)
    lbl = per[:, 4].astype(np.int16)
    lbl[(lbl != 0) & (lbl != 1)] = 255
    lbl[(dyn[:, 2] == ND)] = 255
    elev = a[5 * n].astype(np.float32)
    slope = a[5 * n + 1].astype(np.float32) / 100.0
    order = np.argsort(w["rows"].doy.to_numpy(), kind="stable")
    w["rows"] = w["rows"].iloc[order].reset_index(drop=True)
    return dyn[order], lbl.astype(np.uint8)[order], elev, slope, tr, w["rows"].doy.to_numpy()


def aspect_deg(elev, res=20.0):
    gy, gx = np.gradient(elev, res)
    asp = (np.degrees(np.arctan2(-gx, gy)) + 360) % 360
    return asp


class Modis:
    def __init__(self, path):
        with rasterio.open(path) as d:
            a = d.read()
            self.tr = d.transform
        a = a.astype(np.int16)
        a[a == 255] = -1
        f = a.copy()
        for t in range(1, f.shape[0]):
            m = f[t] < 0
            f[t][m] = f[t - 1][m]
        for t in range(f.shape[0] - 2, -1, -1):
            m = f[t] < 0
            f[t][m] = f[t + 1][m]
        self.fill = f
        del a
        nd = f.shape[0]
        jm = np.arange(1, nd + 1) <= 90
        self.seasonal = (f[jm] == 1).sum(axis=0) >= 10
        free = (f == 0).astype(np.int32)
        run8 = np.lib.stride_tricks.sliding_window_view(free, 8, axis=0).sum(axis=-1) == 8
        run8[:31] = False
        sdd = np.where(run8.any(axis=0), run8.argmax(axis=0) + 1, np.nan).astype(float)
        sdd[~self.seasonal] = np.nan
        self.sdd = sdd

    def cells(self, tr, shape):
        h, w = shape
        xs = tr.c + (np.arange(w) + 0.5) * tr.a
        ys = tr.f + (np.arange(h) + 0.5) * tr.e
        nr, nc = self.fill.shape[1:]
        c = np.clip(np.floor((xs - self.tr.c) / self.tr.a).astype(int), 0, nc - 1)
        r = np.clip(np.floor((ys - self.tr.f) / self.tr.e).astype(int), 0, nr - 1)
        return np.meshgrid(r, c, indexing="ij")

    def day(self, doy, rr, cc):
        return self.fill[doy - 1][rr, cc]


def agg100(ws, valid):
    t, h, w = ws.shape
    h, w = h // 5 * 5, w // 5 * 5
    ws, valid = ws[:, :h, :w], valid[:, :h, :w]
    s = (ws & valid).reshape(t, h // 5, 5, w // 5, 5).sum(axis=(2, 4)).astype(float)
    n = valid.reshape(t, h // 5, 5, w // 5, 5).sum(axis=(2, 4)).astype(float)
    out = np.where(n > 0, (s / np.maximum(n, 1) >= 0.5).astype(float), np.nan)
    return out


def melt_timing(w100, doy):
    pair = (w100[:-1] == 1) & (w100[1:] == 1)
    has = pair.any(axis=0)
    first = pair.argmax(axis=0)
    last = pair.shape[0] - 1 - pair[::-1].argmax(axis=0)
    mo = np.where(has, doy[first], np.nan)
    me = np.where(has, doy[last + 1], np.nan)
    return mo, me


def isolated(ws):
    p = np.pad(ws.astype(np.int8), 1)
    nb = sum(p[1 + di:p.shape[0] - 1 + di, 1 + dj:p.shape[1] - 1 + dj]
             for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0))
    return int((ws & (nb < 2)).sum())


def block_mean(x, k=5):
    h, w = x.shape
    return x[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def stage_prep(a):
    os.makedirs(a.out, exist_ok=True)
    dates = read_dates(a.chips)
    W = windows(a.chips, dates)
    print(f"{len(W)} windows, {len(dates)} acquisition rows")
    mod = Modis(a.modis) if a.modis else None
    lab_rows, cell_rows, ts_rows, sdd_rows = [], [], [], []
    for w in W:
        t0 = time.time()
        dyn, lbl, elev, slope, tr, doy = load(w)
        rc = dyn[:, 2]
        valid = rc != ND
        ws = valid & (rc <= TAU)
        lab_rows.append({"window": w["name"], "block": w["block"], "split": w["split"], "dates": len(doy),
                         "elev_mean": float(np.nanmean(elev)), "valid_px": int(valid.sum()),
                         "ratio_wet_px": int(ws.sum()), "ratio_dry_px": int((valid & ~ws).sum()),
                         "lbl_wet_px": int((lbl == 1).sum()), "lbl_dry_px": int((lbl == 0).sum()),
                         "lbl_ignore_px": int(((lbl == 255) & valid).sum())})
        w100 = agg100(ws, valid)
        mo, me = melt_timing(w100, doy)
        e100, s100 = block_mean(elev), block_mean(slope)
        asp100 = block_mean(aspect_deg(elev))
        ii, jj = np.meshgrid(np.arange(mo.shape[0]), np.arange(mo.shape[1]), indexing="ij")
        x100 = tr.c + (jj + 0.5) * 100.0
        y100 = tr.f - (ii + 0.5) * 100.0
        cell_rows.append(pd.DataFrame({"window": w["name"], "block": w["block"], "split": w["split"],
                                       "x": x100.ravel(), "y": y100.ravel(), "elev": e100.ravel().round(0),
                                       "slope": s100.ravel().round(1), "aspect": asp100.ravel().round(0),
                                       "mo": mo.ravel(), "me": me.ravel()}))
        if mod is not None:
            rr, cc = mod.cells(tr, elev.shape)
            for k, dy in enumerate(doy):
                m = mod.day(int(dy), rr, cc)
                v = valid[k]
                ts_rows.append({"window": w["name"], "split": w["split"], "doy": int(dy),
                                "date": w["rows"].date[k], "rel_orbit": int(w["rows"].rel_orbit[k]),
                                "pass": w["rows"]["pass"][k], "valid_px": int(v.sum()),
                                "sar_wet_px": int(ws[k].sum()),
                                "modis_snow_px": int((v & (m == 1)).sum()),
                                "modis_free_px": int((v & (m == 0)).sum()),
                                "wet_in_snow_px": int((ws[k] & (m == 1)).sum()),
                                "wet_in_free_px": int((ws[k] & (m == 0)).sum())})
            r100 = ((y100 - mod.tr.f) / mod.tr.e).astype(int)
            c100 = ((x100 - mod.tr.c) / mod.tr.a).astype(int)
            df = pd.DataFrame({"r": r100.ravel(), "c": c100.ravel(), "me": me.ravel(), "mo": mo.ravel(),
                               "elev": e100.ravel()})
            g = df.groupby(["r", "c"]).agg(n=("me", "size"), n_me=("me", "count"), me=("me", "median"),
                                           mo=("mo", "median"), elev=("elev", "mean")).reset_index()
            g = g[(g.n >= 20) & (g.n_me >= 13)]
            g["sdd"] = mod.sdd[g.r.to_numpy(), g.c.to_numpy()]
            g["window"] = w["name"]
            sdd_rows.append(g)
        if w["block"] == a.figure_block:
            rc40 = np.where(valid, rc, np.nan).astype(np.float32).reshape(len(doy), 256, 2, 256, 2)
            rc40 = np.nanmean(rc40, axis=(2, 4)) / 100.0
            extra = {}
            if mod is not None:
                k = int(np.argmax(ws.sum(axis=(1, 2))))
                m = mod.day(int(doy[k]), rr, cc)
                cls = np.full(elev.shape, 0, np.uint8)
                cls[(m == 1) & ~ws[k]] = 2
                cls[ws[k] & (m == 1)] = 1
                cls[ws[k] & (m == 0)] = 3
                cls[m < 0] = 4
                cls[~valid[k]] = 5
                extra = {"agree_cls": cls, "agree_doy": int(doy[k])}
            np.savez_compressed(os.path.join(a.out, f"fig_{w['block']}.npz"), rc40=rc40.astype(np.float16),
                                elev40=block_mean(elev, 2).astype(np.float32), doy=doy,
                                date=w["rows"].date.to_numpy().astype(str),
                                orbit=w["rows"].rel_orbit.to_numpy(), transform=np.array(tr)[:6], **extra)
        print(f"  {w['name']:24s} {len(doy):2d} dates  {time.time() - t0:5.1f} s")
    L = pd.DataFrame(lab_rows)
    L.to_csv(os.path.join(a.out, "labels_stats.csv"), index=False)
    C = pd.concat(cell_rows)
    C.to_csv(os.path.join(a.out, "melt_cells_ratio.csv.gz"), index=False, float_format="%.1f")
    summ = {"wet_label_retained_pct": 100 * L.lbl_wet_px.sum() / max(L.ratio_wet_px.sum(), 1),
            "dry_label_retained_pct": 100 * L.lbl_dry_px.sum() / max(L.ratio_dry_px.sum(), 1)}
    ok = C.dropna(subset=["mo"])
    ok = ok[ok.elev >= 1500]
    if len(ok) > 100:
        s = ok.sample(min(len(ok), 20000), random_state=SEED)
        r = stats.theilslopes(s.mo, s.elev)
        summ.update({"cells_with_melt_pct": 100 * C.mo.notna().mean(), "mo_min": ok.mo.quantile(0.02),
                     "mo_max": ok.mo.quantile(0.98), "me_min": ok.me.quantile(0.02), "me_max": ok.me.quantile(0.98),
                     "mo_slope_d100m": 100 * r[0], "mo_slope_ci": [100 * r[2], 100 * r[3]],
                     "mo_elev_r": float(np.corrcoef(s.mo, s.elev)[0, 1])})
    if mod is not None:
        T = pd.DataFrame(ts_rows)
        T.to_csv(os.path.join(a.out, "sar_modis_dates.csv"), index=False)
        S = pd.concat(sdd_rows)
        S.to_csv(os.path.join(a.out, "sar_me_vs_modis_sdd.csv"), index=False, float_format="%.1f")
        S2 = S.dropna(subset=["me", "sdd"])
        summ.update({"inclusion_rate_pct": 100 * T.wet_in_snow_px.sum() / max(T.wet_in_snow_px.sum()
                                                                              + T.wet_in_free_px.sum(), 1),
                     "wet_on_modis_free_pct": 100 * T.wet_in_free_px.sum() / max(T.modis_free_px.sum(), 1),
                     "me_minus_sdd_bias_d": float((S2.me - S2.sdd).mean()),
                     "me_sdd_rmse_d": float(np.sqrt(((S2.me - S2.sdd) ** 2).mean())),
                     "me_sdd_r": float(np.corrcoef(S2.me, S2.sdd)[0, 1]) if len(S2) > 2 else None,
                     "me_sdd_n_cells": len(S2)})
    summ["python"] = platform.python_version()
    with open(os.path.join(a.out, "summary_prep.json"), "w") as fh:
        json.dump(summ, fh, indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


def stage_train(a):
    import torch
    import torch.nn.functional as F
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| torch", torch.__version__)
    os.makedirs(a.out, exist_ok=True)
    dates = read_dates(a.chips)
    W = windows(a.chips, dates)
    data = {}
    for w in W:
        dyn, lbl, elev, slope, tr, doy = load(w)
        data[w["name"]] = (dyn, lbl, elev.astype(np.int16), (slope * 10).astype(np.int16), tr, doy)
    tr_names = [w["name"] for w in W if w["split"] == "train"]
    va_names = [w["name"] for w in W if w["split"] == "val"]
    te_names = [w["name"] for w in W if w["split"] == "test"]
    print(f"windows: {len(tr_names)} train, {len(va_names)} val, {len(te_names)} test")

    acc = [[] for _ in range(6)]
    for n in tr_names:
        dyn, lbl, el, sl, _, _ = data[n]
        v = dyn[:, 2] != ND
        for c in range(4):
            acc[c].append(dyn[:, c][v][::50].astype(np.float32) / 100)
        acc[4].append(el[::8, ::8].ravel().astype(np.float32))
        acc[5].append(sl[::8, ::8].ravel().astype(np.float32) / 10)
    mu = np.array([np.concatenate(x).mean() for x in acc], np.float32)
    sd = np.array([np.concatenate(x).std() + 1e-6 for x in acc], np.float32)

    def make_x(n, k, r0=0, c0=0, size=512):
        dyn, lbl, el, sl, _, _ = data[n]
        sl_ = (slice(r0, r0 + size), slice(c0, c0 + size))
        d = dyn[k][:, sl_[0], sl_[1]].astype(np.float32) / 100
        v = dyn[k, 2][sl_] != ND
        x = np.concatenate([d, el[sl_][None].astype(np.float32), sl[sl_][None].astype(np.float32) / 10])
        x = (x - mu[:, None, None]) / sd[:, None, None]
        x[:, ~v] = 0
        y = lbl[k][sl_].astype(np.int64)
        y[~v] = 255
        return x, y

    samples_tr = [(n, k) for n in tr_names for k in range(len(data[n][5])) if (data[n][1][k] != 255).any()]
    samples_va = [(n, k, r, c) for n in va_names for k in range(len(data[n][5]))
                  for r in (0, 256) for c in (0, 256) if (data[n][1][k][r:r + 256, c:c + 256] != 255).any()]
    print(f"chips: {4 * len(samples_tr)} train (random 256-px crops), {len(samples_va)} val")

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class Gate(nn.Module):
        def __init__(self, fg, fx, fi):
            super().__init__()
            self.wg = nn.Sequential(nn.Conv2d(fg, fi, 1), nn.BatchNorm2d(fi))
            self.wx = nn.Sequential(nn.Conv2d(fx, fi, 1), nn.BatchNorm2d(fi))
            self.psi = nn.Sequential(nn.Conv2d(fi, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

        def forward(self, g, x):
            al = self.psi(F.relu(self.wg(g) + self.wx(x)))
            return x * al, al

    class AttUNet(nn.Module):
        def __init__(self, cin=6, b=32):
            super().__init__()
            f = [b, 2 * b, 4 * b, 8 * b, 16 * b]
            self.enc = nn.ModuleList([block(cin, f[0])] + [block(f[i], f[i + 1]) for i in range(4)])
            self.up = nn.ModuleList([nn.ConvTranspose2d(f[i + 1], f[i], 2, stride=2) for i in range(4)])
            self.gate = nn.ModuleList([Gate(f[i], f[i], max(f[i] // 2, 1)) for i in range(4)])
            self.dec = nn.ModuleList([block(2 * f[i], f[i]) for i in range(4)])
            self.head = nn.Conv2d(f[0], 1, 1)

        def forward(self, x):
            skips = []
            for i, e in enumerate(self.enc):
                x = e(x if i == 0 else F.max_pool2d(x, 2))
                skips.append(x)
            x = skips.pop()
            alphas = []
            for i in range(3, -1, -1):
                g = self.up[i](x)
                s, al = self.gate[i](g, skips[i])
                alphas.append(al)
                x = self.dec[i](torch.cat([g, s], 1))
            return self.head(x), alphas[-1]

    net = AttUNet(b=a.base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)

    def loss_fn(logit, y):
        m = (y != 255).float()
        t = (y == 1).float()
        bce = (F.binary_cross_entropy_with_logits(logit, t, reduction="none") * m).sum() / m.sum().clamp(min=1)
        p = torch.sigmoid(logit) * m
        dice = 1 - (2 * (p * t).sum() + 1e-6) / (p.sum() + (t * m).sum() + 1e-6)
        return 0.5 * bce + 0.5 * dice

    def scores(tp, fp, fn, tn):
        n = tp + fp + fn + tn
        oa = (tp + tn) / max(n, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(n * n, 1)
        kap = (oa - pe) / max(1 - pe, 1e-9)
        return oa, f1, iou, kap

    def counts(pred, y):
        m = y != 255
        p, t = pred[m], y[m] == 1
        return np.array([(p & t).sum(), (p & ~t).sum(), (~p & t).sum(), (~p & ~t).sum()], np.int64)

    hist, best, bad, t_ep = [], 1e9, 0, None
    bs = 16
    if a.eval_only:
        print("evaluation only: loading", os.path.join(a.out, "attunet_best.pt"))
        old = os.path.join(a.out, "training_history.csv")
        hist = pd.read_csv(old).to_dict("records") if os.path.exists(old) else []
        best = min((h["val_loss"] for h in hist), default=float("nan"))
    for ep in range(1, 0 if a.eval_only else a.epochs + 1):
        t0 = time.time()
        net.train()
        idx = [s for s in samples_tr for _ in range(4)]
        np.random.shuffle(idx)
        tl = []
        for i in range(0, len(idx), bs):
            xb, yb = [], []
            for n, k in idx[i:i + bs]:
                r0, c0 = np.random.randint(0, 257, 2)
                x, y = make_x(n, k, r0, c0, 256)
                if np.random.rand() < 0.5:
                    x, y = x[:, :, ::-1], y[:, ::-1]
                if np.random.rand() < 0.5:
                    x, y = x[:, ::-1], y[::-1]
                xb.append(x.copy())
                yb.append(y.copy())
            xb = torch.from_numpy(np.stack(xb)).to(dev)
            yb = torch.from_numpy(np.stack(yb)).to(dev)
            logit, _ = net(xb)
            loss = loss_fn(logit[:, 0], yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl.append(loss.item())
        net.eval()
        vl, cnt = [], np.zeros(4, np.int64)
        with torch.no_grad():
            for i in range(0, len(samples_va), bs):
                xs, ys = zip(*[make_x(n, k, r, c, 256) for n, k, r, c in samples_va[i:i + bs]])
                xb = torch.from_numpy(np.stack(xs)).to(dev)
                yb = torch.from_numpy(np.stack(ys)).to(dev)
                logit, _ = net(xb)
                vl.append(loss_fn(logit[:, 0], yb).item())
                cnt += counts((torch.sigmoid(logit[:, 0]) >= 0.5).cpu().numpy(), np.stack(ys))
        _, f1, iou, _ = scores(*cnt)
        hist.append({"epoch": ep, "train_loss": np.mean(tl), "val_loss": np.mean(vl), "val_f1": f1, "val_iou": iou})
        pd.DataFrame(hist).to_csv(os.path.join(a.out, "training_history.csv"), index=False, float_format="%.5f")
        dt = time.time() - t0
        if t_ep is None:
            t_ep = dt
            print(f"epoch 1 took {dt / 60:.1f} min -> at most {a.epochs * dt / 60:.0f} min for {a.epochs} epochs "
                  f"(early stopping usually ends sooner). Ctrl-C to stop; rerun with --base 16 to halve the time.")
        print(f"epoch {ep:3d}  train {np.mean(tl):.4f}  val {np.mean(vl):.4f}  F1 {f1:.3f}  IoU {iou:.3f}  {dt:.0f} s")
        if np.mean(vl) < best:
            best, bad = np.mean(vl), 0
            torch.save(net.state_dict(), os.path.join(a.out, "attunet_best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("early stopping")
                break

    net.load_state_dict(torch.load(os.path.join(a.out, "attunet_best.pt"), map_location=dev))
    net.eval()
    units, cell_rows, examples = [], [], []
    mod = Modis(a.modis) if a.modis else None
    with torch.no_grad():
        for n in te_names:
            dyn, lbl, el, sl, tr, doy = data[n]
            probs = np.zeros(lbl.shape, np.float32)
            alph = np.zeros(lbl.shape, np.float32)
            for k in range(len(doy)):
                h, w = lbl.shape[1:]
                x, _ = make_x(n, k, 0, 0, max(h, w))
                ph, pw = (-h) % 16, (-w) % 16
                x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
                lo, al = net(torch.from_numpy(x[None]).to(dev))
                probs[k] = torch.sigmoid(lo[0, 0]).cpu().numpy()[:h, :w]
                alph[k] = al[0, 0].cpu().numpy()[:h, :w]
            valid = dyn[:, 2] != ND
            ws_r = valid & (dyn[:, 2] <= TAU)
            ws_u = valid & (probs >= 0.5)
            if mod is not None:
                rr, cc = mod.cells(tr, el.shape)
            for k in range(len(doy)):
                row = {"window": n, "doy": int(doy[k])}
                for meth, p in (("ratio", ws_r[k]), ("unet", ws_u[k])):
                    c = counts(p, lbl[k].astype(np.int64))
                    row.update({f"{meth}_{q}": int(v) for q, v in zip(("tp", "fp", "fn", "tn"), c)})
                    row[f"{meth}_wet"] = int(p.sum())
                    row[f"{meth}_isolated"] = isolated(p)
                    if mod is not None:
                        m = mod.day(int(doy[k]), rr, cc)
                        row[f"{meth}_wet_in_snow"] = int((p & (m == 1)).sum())
                        row[f"{meth}_wet_in_free"] = int((p & (m == 0)).sum())
                        row["modis_free_px"] = int((valid[k] & (m == 0)).sum())
                units.append(row)
                q = [(lbl[k][r:r + 256, c:c + 256] == 1).sum() for r in (0, 256) for c in (0, 256)]
                j = int(np.argmax(q))
                r0, c0 = (j // 2) * 256, (j % 2) * 256
                sl_ = (slice(r0, r0 + 256), slice(c0, c0 + 256))
                ex = {"window": n, "doy": int(doy[k]), "nwet": int(q[j]), "elev": float(el[sl_].mean()),
                      "vh": dyn[k, 1][sl_], "rc": dyn[k, 2][sl_], "ratio": ws_r[k][sl_], "prob": probs[k][sl_],
                      "alpha": alph[k][sl_]}
                if mod is not None:
                    ex["modis"] = mod.day(int(doy[k]), rr[sl_], cc[sl_])
                examples.append(ex)
            for meth, ws in (("ratio", ws_r), ("unet", ws_u)):
                mo, me = melt_timing(agg100(ws, valid), doy)
                cell_rows.append(pd.DataFrame({"window": n, "method": meth, "elev": block_mean(el.astype(float))
                                               .ravel().round(0), "mo": mo.ravel(), "me": me.ravel()}))
    U = pd.DataFrame(units)
    U.to_csv(os.path.join(a.out, "test_units.csv"), index=False)
    pd.concat(cell_rows).to_csv(os.path.join(a.out, "melt_cells_test.csv.gz"), index=False, float_format="%.1f")
    rng = np.random.default_rng(SEED)
    res = {}
    for meth in ("ratio", "unet"):
        c = U[[f"{meth}_{q}" for q in ("tp", "fp", "fn", "tn")]].to_numpy()
        point = scores(*c.sum(axis=0))
        boot = np.array([scores(*c[rng.integers(0, len(c), len(c))].sum(axis=0)) for _ in range(1000)])
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        res[meth] = {k: [round(float(p), 4), round(float(l), 4), round(float(h), 4)]
                     for k, p, l, h in zip(("OA", "F1", "IoU", "kappa"), point, lo, hi)}
        res[meth]["isolated_wet_pct"] = round(100 * U[f"{meth}_isolated"].sum() / max(U[f"{meth}_wet"].sum(), 1), 2)
        if mod is not None:
            wi, wf = U[f"{meth}_wet_in_snow"].sum(), U[f"{meth}_wet_in_free"].sum()
            res[meth]["inclusion_rate_pct"] = round(100 * wi / max(wi + wf, 1), 2)
            res[meth]["wet_on_modis_free_pct"] = round(100 * wf / max(U.modis_free_px.sum(), 1), 3)
    examples.sort(key=lambda e: -e["nwet"])
    pick, seen = [], set()
    for e in examples:
        if (e["window"], e["doy"]) not in seen and len(pick) < 6:
            pick.append(e)
            seen.add((e["window"], e["doy"]))
    pick.sort(key=lambda e: e["elev"])
    np.savez_compressed(os.path.join(a.out, "fig_examples.npz"),
                        **{f"{k}_{i}": np.asarray(v) for i, e in enumerate(pick) for k, v in e.items()})
    res.update({"epochs_run": len(hist), "best_val_loss": float(best), "train_chips": 4 * len(samples_tr),
                "val_chips": len(samples_va), "test_units": len(U), "base_filters": a.base,
                "torch": torch.__version__, "python": platform.python_version(), "device": str(dev),
                "minutes_per_epoch": round(t_ep / 60, 2) if t_ep else None})
    with open(os.path.join(a.out, "summary_train.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "train"])
    ap.add_argument("--chips", default="chips_small")
    ap.add_argument("--modis", default=None, help="auxiliary binary reference raster")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--base", type=int, default=16, help="filters of the first encoder stage")
    ap.add_argument("--figure-block", default="b11")
    ap.add_argument("--eval-only", action="store_true", help="skip optimisation; evaluate the stored checkpoint")
    args = ap.parse_args()
    stage_prep(args) if args.stage == "prep" else stage_train(args)
