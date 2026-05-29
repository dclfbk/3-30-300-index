var canopy = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1");

var cities = {
  'ancona':      [13.52, 43.62],
  'aosta':       [7.32,  45.74],
  'bari':        [16.87, 41.12],
  'bologna':     [11.34, 44.49],
  'bolzano':     [11.35, 46.50],
  'cagliari':    [9.11,  39.22],
  'campobasso':  [14.66, 41.56],
  'catanzaro':   [16.60, 38.91],
  'firenze':     [11.25, 43.77],
  'genova':      [8.93,  44.41],
  'milano':      [9.19,  45.46],
  'napoli':      [14.27, 40.85],
  'palermo':     [13.36, 38.11],
  'perugia':     [12.39, 43.11],
  'potenza':     [15.80, 40.64],
  'roma':        [12.49, 41.89],
  'torino':      [7.68,  45.07],
  'trento':      [11.12, 46.07],
  'trieste':     [13.77, 45.65],
  'venezia':     [12.33, 45.44],
  'verona':      [10.99, 45.44],
};

Object.keys(cities).forEach(function(name) {
  var coords = cities[name];
  var geom = ee.Geometry.Point(coords).buffer(10000);
  
  // mask to tree pixels only (height > 0)
  var treeMask = canopy.gt(0);
  var treeCanopy = canopy.updateMask(treeMask);
  
  var mean = treeCanopy.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: geom,
    scale: 10,
    maxPixels: 1e10
  });
  
  print(name + ' mean tree height (m):', mean);
});