var JOB  = 'chips';
var YEAR = 2025;
var ASSET_ROOT   = 'projects/PROJECT_ID/assets/s1seg';
var DRIVE_FOLDER = 's1seg_out';
var CURVE_ORBITS = [];

var QUEUE = [];

var sf   = require('users/adugnagirma/gee_s1_ard:speckle_filter');
var trf  = require('users/adugnagirma/gee_s1_ard:terrain_flattening');
var util = require('users/adugnagirma/gee_s1_ard:utilities');

var CRS = 'EPSG:CODE';
var X0 = 0, X1 = 0, Y1 = 0, Y0 = 0;
var AOI_LL  = ee.Geometry.Rectangle([0, 0, 0, 0], 'EPSG:4326', false);
var AOI_UTM = ee.Geometry.Rectangle([X0, Y1, X1, Y0], CRS, false);
var INFER_WINDOWS = {
  w1: [0, 0, 0, 0],
  w2: [0, 0, 0, 0]
};
var MELT_START = '-02-01', MELT_END = '-08-16';
var CHIP_START = '-02-15', CHIP_END = '-07-16';
var TAU = -2.0;
var K = 0.5, T1 = 20.0, T2 = 45.0;
var LAPSE = 0.0065;
var N_BLOCKS = 30, CHIP_PX = 512, MAX_CHIP_DATES = 16, BAND_BATCH = 20, SEED = 42;
var D2R = Math.PI / 180;

var DEM  = ee.Image('USGS/SRTMGL1_003').select('elevation');
var LAND = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(0).lt(50);
var ERA5_DAILY   = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR');
var ERA5_MONTHLY = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR');

function tr(s) { return [s, 0, X0, 0, -s, Y0]; }
function utmRect(b) { return ee.Geometry.Rectangle(b, CRS, false); }
function pad(n, w) { var s = String(n); while (s.length < w) { s = '0' + s; } return s; }
function tag(a) { return 'd' + a.d.replace(/-/g, '') + '_o' + pad(a.o, 3); }
function doy(d) {
  var t = new Date(d + 'T00:00:00Z'), s = new Date(d.slice(0, 4) + '-01-01T00:00:00Z');
  return Math.round((t - s) / 86400000) + 1;
}

function exportImage(img, name, region, scale) {
  Export.image.toDrive({
    image: img, description: name, folder: DRIVE_FOLDER, fileNamePrefix: name,
    region: region, crs: CRS, crsTransform: tr(scale), maxPixels: 1e13,
    fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}
  });
}
function exportTable(fc, name, selectors) {
  Export.table.toDrive({collection: fc, description: name, folder: DRIVE_FOLDER,
                        fileNamePrefix: name, fileFormat: 'CSV', selectors: selectors});
}
function rowsTable(rows) {
  return ee.FeatureCollection(rows.map(function (r) { return ee.Feature(null, r); }));
}

function s1Raw(start, end, region) {
  return ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filterDate(start, end).filterBounds(region)
    .select(['VV', 'VH', 'angle']);
}

function s1Processed(start, end, region, orbit, noSpeckle) {
  var col = s1Raw(start, end, region);
  if (orbit) { col = col.filter(ee.Filter.eq('relativeOrbitNumber_start', orbit)); }
  if (!noSpeckle) { col = ee.ImageCollection(sf.MonoTemporal_Filter(col, 7, 'REFINED LEE')); }
  col = trf.slope_correction(col, 'VOLUME', ee.Image('USGS/SRTMGL1_003'), 0);
  return col.map(util.lin_to_db);
}

function acquisitions(start, end, region) {
  var lst = s1Raw(start, end, region)
    .reduceColumns(ee.Reducer.toList(3),
                   ['system:time_start', 'relativeOrbitNumber_start', 'orbitProperties_pass'])
    .get('list').getInfo();
  var seen = {}, out = [];
  lst.forEach(function (x) {
    var d = new Date(x[0]).toISOString().slice(0, 10), key = d + '_' + x[1];
    if (!seen[key]) { seen[key] = true; out.push({d: d, o: x[1], p: x[2]}); }
  });
  out.sort(function (a, b) { return a.d < b.d ? -1 : (a.d > b.d ? 1 : a.o - b.o); });
  return out;
}

function acqImage(proc, a) {
  var d = ee.Date(a.d);
  return proc.filterDate(d, d.advance(1, 'day'))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', a.o)).mosaic();
}

function localIncidence(img) {
  var heading = ee.Terrain.aspect(img.select('angle'))
    .reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI_LL, scale: 1000, maxPixels: 1e9});
  heading = ee.Dictionary(heading).combine({aspect: 0}, false).get('aspect');
  heading = ee.Number(ee.Algorithms.If(ee.Number(heading).gt(180),
                                       ee.Number(heading).subtract(360), heading));
  var theta = img.select('angle').multiply(D2R);
  var slope = ee.Terrain.slope(DEM).multiply(D2R);
  var aspect = ee.Terrain.aspect(DEM);
  var phiS = aspect.where(aspect.gt(180), aspect.subtract(360)).multiply(-D2R);
  var phiR = ee.Image.constant(heading).multiply(D2R).subtract(phiS);
  var alphaR = slope.tan().multiply(phiR.cos()).atan();
  var alphaAz = slope.tan().multiply(phiR.sin()).atan();
  return theta.subtract(alphaR).cos().multiply(alphaAz.cos()).acos().divide(D2R).rename('TLOC');
}

function reference(o) {

  var r = ee.Image(ASSET_ROOT + '/refc_orbit_' + pad(o, 3));
  return r.updateMask(r.neq(-32768)).divide(100);
}

function ratioWet(img, o) {
  var vvvh = img.select(['VV', 'VH']);
  var r = vvvh.subtract(reference(o).select(['VV', 'VH']));
  var tloc = localIncidence(img);
  var w = ee.Image(1).add(ee.Image(T2).subtract(tloc).divide(T2 - T1)).multiply(K).max(K).min(1);
  var rc = r.select('VH').multiply(w).add(r.select('VV').multiply(ee.Image(1).subtract(w))).rename('RC');
  return ee.Image.cat([vvvh, rc, tloc, rc.lte(TAU).rename('WS')]).updateMask(LAND).clip(AOI_LL);
}

function toI16(img) { return img.multiply(100).round().unmask(-32768).toInt16(); }

var ERA_PROJ = ERA5_DAILY.first().projection();
var Z_ERA = DEM.reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 4096})
  .reproject({crs: 'EPSG:4326', scale: 1000})
  .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 1024})
  .reproject(ERA_PROJ);

function era5Label(ws, d) {
  var d0 = ee.Date(d);
  var today = ee.Image(ERA5_DAILY.filterDate(d0, d0.advance(1, 'day')).first());
  var yday  = ee.Image(ERA5_DAILY.filterDate(d0.advance(-1, 'day'), d0).first());
  function tcorr(im) {
    return im.select('temperature_2m_max').resample('bilinear').subtract(273.15)
      .subtract(DEM.subtract(Z_ERA.resample('bilinear')).multiply(LAPSE));
  }
  var tDay = tcorr(today), tTwo = tDay.max(tcorr(yday));
  var sd = today.select('snow_depth').resample('bilinear');
  return ee.Image(255)
    .where(ws.eq(1).and(tTwo.gte(0)).and(sd.gt(0.01)), 1)
    .where(ws.eq(0).and(tDay.lt(-1).or(sd.lte(0.01))), 0)
    .rename('LBL');
}

function blockWindows() {
  var grid = AOI_UTM.coveringGrid(ee.Projection(CRS).atScale(20000));
  var stats = ee.Image.cat([DEM.rename('elev'), LAND.not().rename('water')]);
  var fc = stats.reduceRegions({collection: grid, reducer: ee.Reducer.mean(),
                                scale: 200, crs: CRS, tileScale: 4});
  fc = fc.map(function (f) {
    var c = f.geometry().centroid(1).transform(CRS, 1).coordinates();
    return f.set({
      inside: f.geometry().intersection(AOI_LL, 10).area(10).divide(f.geometry().area(10)),
      cx: ee.Number(X0).add(ee.Number(c.get(0)).subtract(X0).divide(20).round().multiply(20)),
      cy: ee.Number(Y0).subtract(ee.Number(Y0).subtract(c.get(1)).divide(20).round().multiply(20))
    });
  });
  var elig = fc.filter(ee.Filter.gte('elev', 2000)).filter(ee.Filter.lt('water', 0.1))
               .filter(ee.Filter.gt('inside', 0.99));
  var chosen = elig.randomColumn('r', SEED).sort('r').limit(N_BLOCKS).getInfo().features;
  var nTr = Math.round(0.70 * chosen.length), nVa = Math.round(0.15 * chosen.length);
  var half = CHIP_PX * 20 / 2;
  return chosen.map(function (f, i) {
    var p = f.properties;
    return {block: 'b' + pad(i, 2), split: i < nTr ? 'train' : (i < nTr + nVa ? 'val' : 'test'),
            elev: Math.round(p.elev), xmin: p.cx - half, ymin: p.cy - half,
            xmax: p.cx + half, ymax: p.cy + half};
  });
}

function stack20m(proc, acqs, box, withLabel, eraLast) {
  var bands = [], rows = [];
  acqs.forEach(function (a) {
    var x = ratioWet(acqImage(proc, a), a.o), t = tag(a);
    bands.push(toI16(x.select(['VV', 'VH', 'RC', 'TLOC']))
      .rename([t + '_VV', t + '_VH', t + '_RC', t + '_TLOC']));
    if (withLabel) {
      var lbl = (a.d <= eraLast) ? era5Label(x.select('WS'), a.d) : ee.Image(255);
      bands.push(lbl.unmask(255).toInt16().rename(t + '_LBL'));
    }
    rows.push({tag: t, date: a.d, doy: doy(a.d), rel_orbit: a.o, pass: a.p});
  });
  var terr = ee.Terrain.products(DEM);
  bands.push(ee.Image.cat([DEM.rename('ELEV'), terr.select('slope').multiply(100).rename('SLOPE')])
             .round().toInt16());
  return {img: ee.Image.cat(bands).clip(utmRect(box)), rows: rows};
}

function evenly(arr, n) {
  if (arr.length <= n) { return arr; }
  var out = [], step = (arr.length - 1) / (n - 1);
  for (var i = 0; i < n; i++) { out.push(arr[Math.round(i * step)]); }
  return out;
}

function jobScenes() {
  var col = s1Raw('2014-10-01', '2026-10-01', AOI_LL);
  var fc = col.map(function (im) {
    var ang = im.select('angle').reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI_LL,
                                               scale: 2000, maxPixels: 1e9});
    return ee.Feature(null, {
      id: im.get('system:index'),
      date: ee.Date(im.get('system:time_start')).format('YYYY-MM-dd HH:mm'),
      platform: im.get('platform_number'), rel_orbit: im.get('relativeOrbitNumber_start'),
      pass: im.get('orbitProperties_pass'), mean_angle: ang.get('angle')});
  });
  print('Sentinel-1 scenes over the area:', col.size());
  exportTable(fc, 's1_scenes', ['id', 'date', 'platform', 'rel_orbit', 'pass', 'mean_angle']);
}

function jobReference() {
  var proc = ee.ImageCollection([]);
  for (var y = 2015; y <= 2025; y++) {
    proc = proc.merge(s1Processed(y + '-08-01', y + '-10-01', AOI_LL));
  }
  var orbits = ee.List(s1Raw('2015-08-01', '2025-10-01', AOI_LL)
    .aggregate_array('relativeOrbitNumber_start')).distinct().sort().getInfo();
  print('Relative orbits over the area:', orbits);
  orbits.forEach(function (o) {
    var med = proc.filter(ee.Filter.eq('relativeOrbitNumber_start', o)).select(['VV', 'VH']).median();
    Export.image.toAsset({
      image: toI16(med).set('rel_orbit', o), description: 'ref_orbit_' + pad(o, 3),
      assetId: ASSET_ROOT + '/ref_orbit_' + pad(o, 3), region: AOI_UTM, crs: CRS,
      crsTransform: tr(20), maxPixels: 1e13});
  });
}

function jobRefClean() {

  [43, 50, 72, 79, 123, 145, 152, 174].forEach(function (o) {
    var r = ee.Image(ASSET_ROOT + '/ref_orbit_' + pad(o, 3));
    Export.image.toAsset({
      image: r.updateMask(r.neq(-32768)).set('rel_orbit', o),
      description: 'refc_orbit_' + pad(o, 3), assetId: ASSET_ROOT + '/refc_orbit_' + pad(o, 3),
      region: AOI_UTM, crs: CRS, crsTransform: tr(20), maxPixels: 1e13,
      pyramidingPolicy: {'.default': 'mean'}});
  });
}

function jobStatic() {
  var terr = ee.Terrain.products(DEM);
  var img = ee.Image.cat([DEM.rename('ELEV'), terr.select('slope').multiply(100).rename('SLOPE'),
                          terr.select('aspect').multiply(10).rename('ASPECT'), LAND.rename('LAND')])
            .round().toInt16();
  exportImage(img, 'static_100m', AOI_UTM, 100);
}

function jobBlocks() {
  var w = blockWindows();
  print('Training windows:', w);
  exportTable(rowsTable(w), 'blocks', ['block', 'split', 'elev', 'xmin', 'ymin', 'xmax', 'ymax']);
  w.forEach(function (b) {
    Map.addLayer(utmRect([b.xmin, b.ymin, b.xmax, b.ymax]),
                 {color: b.split === 'train' ? 'blue' : (b.split === 'val' ? 'orange' : 'red')}, b.block);
  });
}

function jobPhenology(year) {
  var s = year + MELT_START, e = year + MELT_END;
  var acqs = acquisitions(s, e, AOI_LL), proc = s1Processed(s, e, AOI_LL), rows = [];
  print('Acquisitions ' + s + ' .. ' + e + ':', acqs.length);
  for (var b0 = 0; b0 < acqs.length; b0 += BAND_BATCH) {
    var part = acqs.slice(b0, b0 + BAND_BATCH);
    var fname = 'phen_' + year + '_part' + pad(b0 / BAND_BATCH + 1, 2);
    var bands = part.map(function (a, i) {
      var ws = ratioWet(acqImage(proc, a), a.o).select('WS');
      var frac = ws.reproject({crs: CRS, crsTransform: tr(20)})
        .reduceResolution({reducer: ee.Reducer.mean(), maxPixels: 64})
        .reproject({crs: CRS, crsTransform: tr(100)});
      rows.push({file: fname, band: i + 1, name: tag(a), date: a.d, doy: doy(a.d),
                 rel_orbit: a.o, pass: a.p});
      return frac.gte(0.5).unmask(255).toUint8().rename(tag(a));
    });
    exportImage(ee.Image.cat(bands), fname, AOI_UTM, 100);
  }
  exportTable(rowsTable(rows), 'phen_' + year + '_bands',
              ['file', 'band', 'name', 'date', 'doy', 'rel_orbit', 'pass']);
}

function jobCurves(year) {

  var s = year + '-01-01', e = year + '-10-01';
  var all = acquisitions(s, e, AOI_LL), orbits = [];
  all.forEach(function (a) { if (orbits.indexOf(a.o) < 0) { orbits.push(a.o); } });
  var zone = DEM.subtract(1200).divide(400).floor().clamp(0, 7).toInt().rename('zone');
  var sel = ['date', 'doy', 'rel_orbit', 'pass'];
  for (var z = 0; z < 8; z++) { sel.push('rvv_z' + z, 'rvh_z' + z, 'rc_z' + z); }
  if (CURVE_ORBITS.length) { orbits = orbits.filter(function (o) { return CURVE_ORBITS.indexOf(o) >= 0; }); }
  orbits.forEach(function (o) {
    var acqs = all.filter(function (a) { return a.o === o; });
    var proc = s1Processed(s, e, AOI_LL, o, true);
    var feats = acqs.map(function (a) {
      var x = ratioWet(acqImage(proc, a), a.o);
      var r = x.select(['VV', 'VH']).subtract(reference(a.o)).rename(['RVV', 'RVH']);
      var grp = ee.Image.cat([r, x.select('RC'), zone]).reduceRegion({
        reducer: ee.Reducer.mean().repeat(3).group({groupField: 3, groupName: 'zone'}),
        geometry: AOI_LL, scale: 300, crs: CRS, maxPixels: 1e10, tileScale: 16});
      var props = ee.Dictionary(ee.List(grp.get('groups')).iterate(function (g, acc) {
        g = ee.Dictionary(g);
        var zz = ee.Number(g.get('zone')).format('%d'), m = ee.List(g.get('mean'));

        var v = function (i) {
          var x = m.get(i);
          return ee.Algorithms.If(ee.Algorithms.IsEqual(x, null), -9999, x);
        };
        return ee.Dictionary(acc).set(ee.String('rvv_z').cat(zz), v(0))
          .set(ee.String('rvh_z').cat(zz), v(1)).set(ee.String('rc_z').cat(zz), v(2));
      }, ee.Dictionary({})));
      return ee.Feature(null, props.combine({date: a.d, doy: doy(a.d), rel_orbit: a.o, pass: a.p}));
    });
    print('curves orbit ' + o + ':', acqs.length + ' acquisitions');
    exportTable(ee.FeatureCollection(feats), 'curves_' + year + '_o' + pad(o, 3), sel);
  });
}

function jobChips(year) {
  var s = year + CHIP_START, e = year + CHIP_END;
  var proc = s1Processed(s, e, AOI_LL), rows = [];
  var eraLast = new Date(ERA5_DAILY.aggregate_max('system:time_start').getInfo())
                  .toISOString().slice(0, 10);
  print('Last ERA5-Land day:', eraLast);
  blockWindows().forEach(function (w) {
    var box = [w.xmin, w.ymin, w.xmax, w.ymax];
    var acqs = evenly(acquisitions(s, e, utmRect(box)), MAX_CHIP_DATES);
    var st = stack20m(proc, acqs, box, true, eraLast);
    var name = 'chips_' + year + '_' + w.block + '_' + w.split;
    exportImage(st.img, name, utmRect(box), 20);
    st.rows.forEach(function (r) { r.file = name; r.block = w.block; r.split = w.split; rows.push(r); });
  });
  exportTable(rowsTable(rows), 'chips_' + year + '_dates',
              ['file', 'block', 'split', 'tag', 'date', 'doy', 'rel_orbit', 'pass']);
}

function jobInfer(year) {
  var s = year + MELT_START, e = year + MELT_END;
  var proc = s1Processed(s, e, AOI_LL), rows = [], step = BAND_BATCH / 2;
  Object.keys(INFER_WINDOWS).forEach(function (wname) {
    var box = INFER_WINDOWS[wname];
    var acqs = acquisitions(s, e, utmRect(box));
    print(wname + ' acquisitions:', acqs.length);
    for (var b0 = 0; b0 < acqs.length; b0 += step) {
      var st = stack20m(proc, acqs.slice(b0, b0 + step), box, false, null);
      var name = 'infer_' + year + '_' + wname + '_part' + pad(b0 / step + 1, 2);
      exportImage(st.img, name, utmRect(box), 20);
      st.rows.forEach(function (r) { r.file = name; r.window = wname; rows.push(r); });
    }
  });
  exportTable(rowsTable(rows), 'infer_' + year + '_dates',
              ['file', 'window', 'tag', 'date', 'doy', 'rel_orbit', 'pass']);
}

function jobModis(year) {
  var d0 = ee.Date(year + '-01-01');
  var n = ee.Date(year + '-10-01').difference(d0, 'day');
  function snowClass(im) {
    var ndsi = im.select('NDSI_Snow_Cover');
    return ndsi.gte(40).updateMask(ndsi.lte(100)).rename('snow')
      .copyProperties(im, ['system:time_start']);
  }
  var both = ee.ImageCollection('MODIS/061/MOD10A1').merge(ee.ImageCollection('MODIS/061/MYD10A1'))
    .filterDate(d0.advance(-1, 'day'), d0.advance(n.add(1), 'day')).map(snowClass);
  var empty = ee.ImageCollection([ee.Image.constant(0).updateMask(0).rename('snow')]);
  var daily = ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(function (i) {
    var d = d0.advance(i, 'day');
    var day = both.filterDate(d, d.advance(1, 'day')).merge(empty).max();
    var near = both.filterDate(d.advance(-1, 'day'), d.advance(2, 'day')).merge(empty).max();
    return day.unmask(near).unmask(255).toUint8()
      .rename(ee.String('d').cat(d.format('YYYYMMdd')));
  }));
  var stack = daily.toBands();
  stack = stack.rename(stack.bandNames().map(function (b) { return ee.String(b).split('_').get(1); }));
  exportImage(stack.clip(AOI_LL).unmask(255).toUint8(), 'modis_snow_' + year, AOI_UTM, 500);
}

function jobEra5() {
  var high = DEM.gte(2000);
  var fc = ERA5_MONTHLY.filterDate('2014-10-01', '2026-10-01').map(function (im) {
    var t = im.select('temperature_2m').subtract(273.15);
    var all = t.reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI_LL, scale: 11132, maxPixels: 1e9});
    var hi = t.updateMask(high).reduceRegion({reducer: ee.Reducer.mean(), geometry: AOI_LL,
                                             scale: 1000, maxPixels: 1e10, tileScale: 4});
    return ee.Feature(null, {month: ee.Date(im.get('system:time_start')).format('YYYY-MM'),
                             t2m_mean_aoi: all.get('temperature_2m'),
                             t2m_mean_above2000m: hi.get('temperature_2m')});
  });
  exportTable(fc, 'era5_monthly_t2m', ['month', 't2m_mean_aoi', 't2m_mean_above2000m']);
}

Map.centerObject(AOI_LL, 7);
Map.addLayer(AOI_LL, {color: 'yellow'}, 'AOI', true, 0.3);
var JOBS = {scenes: jobScenes, reference: jobReference, refclean: jobRefClean, 'static': jobStatic, blocks: jobBlocks,
            phenology: jobPhenology, curves: jobCurves, chips: jobChips, infer: jobInfer,
            modis: jobModis, era5: jobEra5};
if (QUEUE.length) {
  QUEUE.forEach(function (q) { print('Queue -> job:', q[0], 'year:', q[1]); JOBS[q[0]](q[1]); });
} else {
  print('Job:', JOB, 'Year:', YEAR);
  JOBS[JOB](YEAR);
}
print('Done. Open the Tasks tab and click RUN on each task.');
