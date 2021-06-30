import os
import numpy as np
from shapely.geometry import Polygon
import csv
import pandas as pd
import cv2
import json

DIM_PIXELS = 256  # The width and height value at zoom level 13
DIM_METERS = 107.52  # The width and height value at zoom level 13
ZOOM_LEVEL = 13
PIXEL_IN_METERS = DIM_METERS / DIM_PIXELS


def _get_polygon_center(polygon_tuple):
    """ Get center of a polygon in pixels using Shapely """
    return Polygon(polygon_tuple).centroid


def _get_left_below_tile_coordinates(tile_x_y):
    """
    Return left below Rijksdriehoek coordinates of tile in meters.
    Tile structure info:
    https://www.geonovum.nl/uploads/standards/downloads/nederlandse_richtlijn_tiling_-_versie_1.1.pdf
    """
    # Convert string to int
    X, Y = [int(s) for s in tile_x_y.split("/")]

    # The following values come from pdf
    t = (903401.92 - 22598.08) * 0.5**ZOOM_LEVEL  # Tile width in meters
    tile_x = X * t - 285401.92
    tile_y = Y * t + 22598.08

    return tile_x, tile_y


def _get_rijksdriehoek_coordinates(tile_coordinates, x_coord, y_coord):
    """ Convert the pixel coordinates to Rijksdriehoek coordinates """
    rd_x = tile_coordinates[0] + (PIXEL_IN_METERS * x_coord)
    rd_y = tile_coordinates[1] + (PIXEL_IN_METERS * (DIM_PIXELS - y_coord))
    return [round(rd_x, 3), round(rd_y, 3)]


def _minimum_area_rectangle(polygon_tuple):
    """ Rotated minumum bounding rectangle """
    points = np.array(polygon_tuple, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    width, length = rect[1]

    # Convert pixels to meters
    width_meters = width * PIXEL_IN_METERS
    length_meters = length * PIXEL_IN_METERS

    return round(width_meters, 3), round(length_meters, 3)


def segmentations_to_coordinates(in_file, out_file):
    """
    Get the dimensions of a detected houseboat in meters and get the
    Rijksdriehoek coordinates of the center of the detected houseboat.
    """

    rows_list = []

    # Iterate over csv
    with open(in_file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)  # skip the first line
        for row in csvreader:
            if len(row) != 2:
                print("Broken entry ignored")
                continue

            polygon_data = {}

            # Read the two columns
            polygon_data["tile_x_y"] = row[0]
            polygon_data["mask"] = eval(row[1])

            # Get the center of a polygon in Rijksdriehoek coordinates
            instance_center = _get_polygon_center(polygon_data["mask"])
            tile_coordinates = _get_left_below_tile_coordinates(polygon_data["tile_x_y"])
            polygon_data["center_mask"] = _get_rijksdriehoek_coordinates(tile_coordinates,
                                                                         instance_center.x, instance_center.y)

            # Get width and length of polygon using rotated minumum bounding rectangle
            polygon_data["width"], polygon_data["length"] = _minimum_area_rectangle(polygon_data["mask"])

            rows_list.append(polygon_data)

    # Save this file
    df = pd.DataFrame(rows_list)
    df.to_csv(out_file, index=False)

def data_to_geojson(in_file, out_file):
    """ Polygon data from csv file to geosjon format. """
    geos = []

    # Iterate over csv
    with open(in_file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)  # skip the first line
        for row in csvreader:
            # Read the polygon columns
            polygon = eval(row[1])

            rd_polygon_list = []
            tile_coordinates = _get_left_below_tile_coordinates(row[0])
            for xy in polygon:
                xy_rd = _get_rijksdriehoek_coordinates(tile_coordinates, xy[0], xy[1])
                rd_polygon_list.append(xy_rd)


            poly = {
                'type': 'Polygon',
                'coordinates': [rd_polygon_list]
            }

            geometry = {
                'geometry': poly
            }

            geos.append(geometry)


    geojson_dict = {
        'type': 'FeatureCollection',
        'features': geos,
    }

    # Save the geosjon file
    with open(out_file, 'w') as f:
        json.dump(geojson_dict, f)


