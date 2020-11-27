# Some basic setup
# Import some common libraries
import numpy as np
import cv2
import zipfile
import pandas as pd
import os
from shapely.geometry import Polygon
from skimage import measure

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# Import some common detectron2 utilities
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

def mask_to_simplified_polygon(mask):
    """
    Find contours (boundary lines) around the mask and
    convert it to MS COCO polygon representation.
    """
    contours = measure.find_contours(mask, 0.5, positive_orientation='low')

    segmentations = []
    for contour in contours:
        if len(contour) < 3:
            # Polygons must have at least three points
            continue

        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        if poly.is_empty:
            # Go to next iteration, dont save empty values in list
            continue

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return segmentations

def main():
    """
    An example script on how to iterate over the images in a zip file
    and get predictions from Mask R-CNN.
    """

    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.OUTPUT_DIR = "output"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # Houseboat
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95

    predictor = DefaultPredictor(cfg)

    zip_file = zipfile.ZipFile("Datasets/houseboats/test/images.zip")

    rows_list = []
    for name in zip_file.namelist():
        if name.endswith(".jpeg"):
            filename = name.split("/")[-1].split(".jpeg")[0]

            # Open the images with the openCV reader because BGR order is used in Detectron2
            pic = zip_file.read(name)
            im = cv2.imdecode(np.frombuffer(pic, np.uint8), 1)

            # Use defaultPredicter
            outputs = predictor(im)

            all_instances = outputs["instances"].to("cpu")

            if all_instances.has("pred_masks"):
                masks = all_instances.pred_masks

                for instance in masks:
                    # Convert the binary mask to a polygon (list of points of the polygon)
                    polygon = mask_to_simplified_polygon(instance)

                    if polygon:
                        # Replace negative values with 0
                        polygon_non_negative = [0 if i < 0 else i for i in polygon[0]]

                        # Convert list into list of tuples of every two elements
                        polygon_tuple = list(zip(polygon_non_negative[::2], polygon_non_negative[1::2]))

                        new_data = {"tile_z_x_y" : filename, "mask" : polygon_tuple}
                        rows_list.append(new_data)

    # Save this file
    df_output = pd.DataFrame(rows_list)
    compression_opts = dict(method="zip", archive_name="predicted_houseboats.csv")
    df_output.to_csv("predicted_houseboats.zip", index=False, compression=compression_opts)


if __name__ == "__main__":
    main()
