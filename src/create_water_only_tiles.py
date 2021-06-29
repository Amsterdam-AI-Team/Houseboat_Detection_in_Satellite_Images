import glob
import os
import cv2
import numpy as np

# Width (256) * height (256)
pixels_in_image = 65536

# Define the HSV lower and upper range for masking the image
lower_color = np.array([60, 3, 183])
upper_color = np.array([97, 90, 216])

# Define the thresholds
thresh_perc_black = 0.97
thresh_perc_white = 0.30

# Define RGB colors
COLOR_BLACK = [0, 0, 0]
COLOR_WHITE = [255, 255, 255]

def create_water_only_tiles(in_folder_lufo, in_folder_topo, out_folder):
    """ Filter out bodies of water, because houseboats will (almost) always occur on water. """
    for lufo_filepath in glob.glob(in_folder_lufo + '**/*.jpeg', recursive=True):
        base_filepath = os.path.splitext(lufo_filepath.split(in_folder_lufo)[-1])[0]

        topo_filepath = in_folder_topo + base_filepath + ".png"

        # Read the image from both topo and lufo
        img_lufo = cv2.imread(lufo_filepath)
        img_topo = cv2.imread(topo_filepath)

        # HSV (hue, saturation, value) colorspace
        hsv = cv2.cvtColor(img_topo, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # If the mask is almost completely black, skip
        if (mask == 0).mean() < thresh_perc_black:
            # Calculate the number of white pixels in the lufo image
            # These are (partly) captured lufo images
            img2gray_lufo = cv2.cvtColor(img_lufo, cv2.COLOR_BGR2GRAY)
            perc_white_pixels = np.sum(img2gray_lufo == 255) / pixels_in_image

            # One image has 65536 pixels
            if perc_white_pixels < thresh_perc_white:
                mask = mask / 255
                mask = mask.astype(np.bool)

                # Create a black and white mask image
                res = img_topo.copy()
                res[:, :, :3][~mask] = COLOR_BLACK
                res[:, :, :3][mask] = COLOR_WHITE

                # Overlay the black and white mask image with the satellite image
                img2gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                img_overlay = cv2.bitwise_and(img_lufo, img_lufo, mask=img2gray)            

                # Save the overlayed image
                save_path = os.path.join(out_folder, base_filepath + ".jpeg")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, img_overlay)
