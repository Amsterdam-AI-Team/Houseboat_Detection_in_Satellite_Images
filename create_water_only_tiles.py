import os
import numpy as np
import zipfile
import cv2

COLOR_BLACK = [0, 0, 0]
COLOR_WHITE = [255, 255, 255]
IMAGES_OUTPUT = "satellite_masks/"
ZOOM_LEVEL = "13"

# Zip files with X folder and Y file structure, similar to the structure on:
# https://t1.data.amsterdam.nl/topo_RD/13/
ZIP_FILE_LUFO = zipfile.ZipFile("images_lufo_2020_water_only.zip")
ZIP_FILE_TOPO = zipfile.ZipFile("images_topo_2020.zip")

def main():
    """ Overlay the satellite images with a non-water mask image """

    # Create a directory, first check if it already exists
    if not os.path.exists(IMAGES_OUTPUT):
        os.makedirs(IMAGES_OUTPUT)
    else:
        print("Output folder already exists.")

    for name in ZIP_FILE_LUFO.namelist():
        if name.endswith(".jpeg"):
            filename = name.split("/")[-1].split(".jpeg")[0]
            topo_filename = name.split(".jpeg")[0] + ".png"

            pic = ZIP_FILE_LUFO.read(name)
            img_satellite = cv2.imdecode(np.frombuffer(pic, np.uint8), 1)

            pic = ZIP_FILE_TOPO.read(topo_filename)
            im = cv2.imdecode(np.frombuffer(pic, np.uint8), 1)

            res = im.copy()
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            # Define the lower and upper range for masking the image
            lower_color = np.array([60, 3, 183])
            upper_color = np.array([97, 90, 216])
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # If the mask is almost completely black, skip
            if (mask == 0).mean() < 0.97:
                mask = mask / 255
                mask = mask.astype(np.bool)

                # Create a black and white mask image
                res[:, :, :3][~mask] = COLOR_BLACK
                res[:, :, :3][mask] = COLOR_WHITE

                # Overlay the black and white mask image with the satellite image
                img2gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
                img_overlay = cv2.bitwise_and(img_satellite, img_satellite, mask=img2gray)

                # Calculate the number of white pixels in the satellite image
                # These are (partly) invalid satellite images
                img2gray_sat = cv2.cvtColor(img_satellite, cv2.COLOR_BGR2GRAY)
                n_white_pixels = np.sum(img2gray_sat == 255)

                # One image has 65536 pixels. If 20000 white pixels or more found, skip
                if n_white_pixels < 20000:
                    # For example an output file with name "13_3700_4220.jpeg"
                    cv2.imwrite(os.path.join(IMAGES_OUTPUT, ZOOM_LEVEL + "_" + name.split("/")[0] + "_" + filename + ".jpeg"), img_overlay)

if __name__ == "__main__":
    main()
