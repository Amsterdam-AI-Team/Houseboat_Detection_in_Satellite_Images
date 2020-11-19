## Introduction
TODO

## Map tile structure
Information regarding the tile structure used at [data.amsterdam.nl](https://data.amsterdam.nl) is available here: https://www.geonovum.nl/uploads/standards/downloads/nederlandse_richtlijn_tiling_-_versie_1.1.pdf

## Annotations
The Key register Addresses and Buildings (in Dutch: Basisregistratie Adressen en Gebouwen, BAG) is a collection of base information about addresses and buildings in the Netherlands. The location and size of registered houseboats are also available in a topographic map of BAG. In order to filter for houseboats in the image, we set a boundary as to which color needs to be detected. This results in black/white mask images. Next, we transform the mask images to MS COCO annotations (JSON files) using [this](https://github.com/chrise96/image-to-coco-json-converter) tool.

| ![Topographic map](./media/3810_4315_topo.png) | ![Mask image](./media/3810_4315_mask.png)|![Annotations](./media/3810_4315_detections.jpeg) |
|:---:|:---:|:---:|

## Houseboat detection
Faster R-CNN is used for the detection of houseboats in satellite images. To simplify the detection of houseboats, we overlay the satellite images with a non-water mask image. Using the mask, we avoid the detection of houses on land.

| ![Topographic map](./media/3810_4315_topo.png) | ![Satellite image](./media/3810_4315_lufo.jpeg)|![Water only](./media/3810_4315_mask.jpeg) |
|:---:|:---:|:---:|