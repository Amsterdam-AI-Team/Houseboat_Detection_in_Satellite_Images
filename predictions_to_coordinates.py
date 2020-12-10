"""
Get the dimensions of a detected houseboat in meters and get the
Rijksdriehoek coordinates of the center of the detected houseboat.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import csv
from collections import defaultdict
import time
import pandas as pd
import cv2

DIM_PIXELS = 256 # The width and height value at zoom level 13
DIM_METERS = 107.52 # The width and height value at zoom level 13
PIXEL_IN_METERS = DIM_METERS / DIM_PIXELS
MASK_IMAGES_OUTPUT = "output_mask"
CSV_OUTPUT = "output"
INPUT_FILE = "models/output/predicted_houseboats.csv"
DEBUG = False

def get_polygon_center(polygon_tuple):
    """ Get center of a polygon in pixels using Shapely """
    return Polygon(polygon_tuple).centroid

def get_left_below_tile_coordinates(tile_z_x_y):
    """
    Return left below Rijksdriehoek coordinates of tile in meters.
    Tile structure info:
    https://www.geonovum.nl/uploads/standards/downloads/nederlandse_richtlijn_tiling_-_versie_1.1.pdf
    """
    # Convert string to int
    Z, X, Y = [int(s) for s in tile_z_x_y.split("_")]

    # The following values come from "nederlandse_richtlijn_tiling_-_versie_1.1.pdf"
    t = (903401.92 - 22598.08) * 0.5**Z  # Tile width in meters
    tile_x = X * t - 285401.92
    tile_y = Y * t + 22598.08

    return tile_x, tile_y

def get_rijksdriehoek_coordinates(tile_coordinates, instance_center):
    """ Convert the pixel coordinates to Rijksdriehoek coordinates """
    instance_x = tile_coordinates[0] + (PIXEL_IN_METERS * instance_center.x)
    instance_y = tile_coordinates[1] + (PIXEL_IN_METERS * (256 - instance_center.y))
    return [round(instance_x, 3), round(instance_y, 3)]

def minimum_area_rectangle(polygon_tuple):
    """ Rotated minumum bounding rectangle """
    points = np.array(polygon_tuple, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    width, length = rect[1]

    # Convert pixels to meters
    width_meters = width * PIXEL_IN_METERS
    length_meters = length * PIXEL_IN_METERS

    return round(width_meters, 3), round(length_meters, 3)

def group_by_tile(input_file):
    """ Group csv data by column 'tile_z_x_y' (also known as the filename) """
    grouped_tiles = defaultdict(list)  # each entry of the dict is, by default, an empty list

    with open(input_file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)  # skip the first header line
        for row in csvreader:
            grouped_tiles[row[0]].append(row[1])

    return grouped_tiles

def draw_binary_mask(grouped_tiles):
    """ Draw binary polygon mask of detected instances """
    for key in grouped_tiles:
        # Create a grayscale image
        img = Image.new('L', (DIM_PIXELS, DIM_PIXELS), 0)
        # Loop over polygon masks
        for polygon in grouped_tiles[key]:
            # Convert list enclosed within string to list
            list_polygon = eval(polygon)
            # Fill polygon
            ImageDraw.Draw(img).polygon(list_polygon, outline=0, fill=1)
        mask = np.array(img)
        # Save binary mask images
        Image.fromarray((mask*255).astype(np.uint8)).save(os.path.join(MASK_IMAGES_OUTPUT, key + ".png"))

def main():
    """ Main code """
    start = time.time()

    # Create a directory, first check if it already exists
    if not os.path.exists(CSV_OUTPUT):
        os.makedirs(CSV_OUTPUT)
    else:
        print("Output folder already exists.")

    rows_list = []

    # Iterate over csv
    with open(INPUT_FILE, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)  # skip the first line
        for row in csvreader:
            if len(row) != 2:
                print("Broken entry ignored")
                continue

            polygon_data = {}

            # Read the two columns
            polygon_data["tile_z_x_y"] = row[0]
            polygon_data["mask"] = eval(row[1])

            # Get the center of a polygon in Rijksdriehoek coordinates
            instance_center = get_polygon_center(polygon_data["mask"])
            tile_coordinates = get_left_below_tile_coordinates(polygon_data["tile_z_x_y"])
            polygon_data["center_mask"] = get_rijksdriehoek_coordinates(tile_coordinates, instance_center)

            # Get width and length of polygon using rotated minumum bounding rectangle
            polygon_data["width"], polygon_data["length"] = minimum_area_rectangle(polygon_data["mask"])

            rows_list.append(polygon_data)

    if DEBUG:
        # Create a directory, first check if it already exists
        if not os.path.exists(MASK_IMAGES_OUTPUT):
            os.makedirs(MASK_IMAGES_OUTPUT)
        else:
            print("Output folder already exists.")

        # Draw binary masks to visually validate the quality of the predictions
        grouped_tiles = group_by_tile(INPUT_FILE)
        draw_binary_mask(grouped_tiles)

    # Save this file
    df = pd.DataFrame(rows_list)
    compression_opts = dict(method="zip", archive_name="houseboat_polygon_data.csv")
    df.to_csv(os.path.join(CSV_OUTPUT, "houseboat_polygon_data.zip"), index=False, compression=compression_opts)

    print("Elapsed total time: {0:.2f} seconds.".format(time.time() - start))

if __name__ == "__main__":
    main()
