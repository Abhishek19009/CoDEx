import json
import numpy as np
from os.path import join

# Earth's radius in km
R = 6371  

# Convert lat/lon to local Cartesian coordinates
def latlon_to_cartesian(lat, lon, lat0, lon0):
    x = R * (np.radians(lon) - np.radians(lon0)) * np.cos(np.radians(lat0))
    y = R * (np.radians(lat) - np.radians(lat0))
    return np.column_stack((x, y))  # Shape: (N, 2)


path = './datasets/DynamicEarthNet'
gps_coord = json.load(open(join(path, "coord.json"), "r"))
multihead_ids = json.load(open(join(path, "split.json")))["train"]

latlon_values = np.array([gps_coord[key] for key in multihead_ids])
latitudes, longitudes = latlon_values[:, 0], latlon_values[:, 1]

# Reference point (first coordinate)
lat0, lon0 = latitudes[0], longitudes[0]

cartesion_coords = latlon_to_cartesian(latitudes, longitudes, lat0, lon0)
angles = np.arctan2(cartesion_coords[:, 1], cartesion_coords[:, 0])

# angle diff matrix
angle_diff_matrix = angles[:, np.newaxis] - angles[np.newaxis, :] # Shape: (N, N)