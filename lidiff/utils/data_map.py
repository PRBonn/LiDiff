learning_map = {
  0 : 0,     # "unlabeled"
  1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
  10: 1,     # "car"
  11: 2,     # "bicycle"
  13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
  15: 3,     # "motorcycle"
  16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
  18: 4,     # "truck"
  20: 5,     # "other-vehicle"
  30: 6,     # "person"
  31: 7,     # "bicyclist"
  32: 8,     # "motorcyclist"
  40: 9,     # "road"
  44: 10,    # "parking"
  48: 11,    # "sidewalk"
  49: 12,    # "other-ground"
  50: 13,    # "building"
  51: 14,    # "fence"
  52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
  60: 9,     # "lane-marking" to "road" ---------------------------------mapped
  70: 15,    # "vegetation"
  71: 16,    # "trunk"
  72: 17,    # "terrain"
  80: 18,    # "pole"
  81: 19,    # "traffic-sign"
  99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
  252: 1,    # "moving-car"
  253: 7,    # "moving-bicyclist"
  254: 6,    # "moving-person"
  255: 8,    # "moving-motorcyclist"
  256: 5,    # "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
  257: 5,    # "moving-bus" mapped to "moving-other-vehicle" -----------mapped
  258: 4,    # "moving-truck"
  259: 5,    # "moving-other-vehicle"
}

content = { # as a ratio with the total number of points
  0: 0.03150183342534689,
  1: 0.042607828674502385,
  2: 0.00016609538710764618,
  3: 0.00039838616015114444,
  4: 0.0021649398241338114,
  5: 0.0018070552978863615,
  6: 0.0003375832743104974,
  7: 0.00012711105887399155,
  8: 3.746106399997359e-05,
  9: 0.19879647126983288,
  10: 0.014717169549888214,
  11: 0.14392298360372,
  12: 0.0039048553037472045,
  13: 0.1326861944777486,
  14: 0.0723592229456223,
  15: 0.26681502148037506,
  16: 0.006035012012626033,
  17: 0.07814222006271769,
  18: 0.002855498193863172,
  19: 0.0006155958086189918,
}

content_indoor = {
  0: 0.18111755628849344,
  1: 0.15350115272756307,
  2: 0.264323444618407,
  3: 0.017095487624768667,
  4: 0.02018415055214108,
  5: 0.025684283218171625,
  6: 0.05237503359636922,
  7: 0.03495118545614923,
  8: 0.04252921527371275,
  9: 0.004767541066020183,
  10: 0.06899976905686542,
  11: 0.012345517150886037,
  12: 0.12212566337045223,
}

labels = {
  0: "unlabeled",
  1: "car",
  2: "bicycle",
  3: "motorcycle",
  4: "truck",
  5: "other-vehicle",
  6: "person",
  7: "bicyclist",
  8: "motorcyclist",
  9: "road",
  10: "parking",
  11: "sidewalk",
  12: "other-ground",
  13: "building",
  14: "fence",
  15: "vegetation",
  16: "trunk",
  17: "terrain",
  18: "pole",
  19: "traffic-sign",
}

color_map = {
  0: [0, 0, 0],
  1: [245, 150, 100],
  2: [245, 230, 100],
  3: [150, 60, 30],
  4: [180, 30, 80],
  5: [255, 0, 0],
  6: [30, 30, 255],
  7: [200, 40, 255],
  8: [90, 30, 150],
  9: [255, 0, 255],
  10: [255, 150, 255],
  11: [75, 0, 75],
  12: [75, 0, 175],
  13: [0, 200, 255],
  14: [50, 120, 255],
  15: [0, 175, 0],
  16: [0, 60, 135],
  17: [80, 240, 150],
  18: [150, 240, 255],
  19: [0, 0, 255],
}
