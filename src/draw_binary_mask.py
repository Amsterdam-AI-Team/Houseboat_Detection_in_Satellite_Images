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

def group_by_tile(input_file):
    """ Group csv data by column 'tile_z_x_y' (also known as the filename) """
    grouped_tiles = defaultdict(list)  # each entry of the dict is, by default, an empty list

    with open(input_file, "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csvfile)  # skip the first header line
        for row in csvreader:
            grouped_tiles[row[0]].append(row[1])

    return grouped_tiles

def draw_binary_mask(in_file, out_folder_masks):
    """ Draw binary polygon mask of detected instances """

    # Create a directory, first check if it already exists
    if not os.path.exists(out_folder_masks):
        os.makedirs(out_folder_masks)
    else:
        print("Output folder already exists.")

    # Draw binary masks to validate the quality of the predictions
    grouped_tiles = group_by_tile(in_file)

    for key in grouped_tiles:
        # Create a new image with a white background
        img = Image.new('RGB', (DIM_PIXELS, DIM_PIXELS), (255,255,255))
        # Loop over polygon masks
        for polygon in grouped_tiles[key]:
            # Convert list enclosed within string to list
            list_polygon = eval(polygon)
            # Fill polygon
            ImageDraw.Draw(img).polygon(list_polygon, outline=0, fill=(0, 0, 0))
        mask = np.array(img)

        # Save binary mask images
        save_path = os.path.join(out_folder_masks, key + ".png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray((mask).astype(np.uint8)).save(save_path)
