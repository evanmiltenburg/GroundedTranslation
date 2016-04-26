# MS COCO negation images

This folder contains all scripts used to generate the test set of MS COCO images
with descriptions containing negations. This is what you need to do to reproduce
this set. Note that the relevant `dataset.h5` file is included in this repository, so you could just use that to generate descriptions for the MS COCO images with descriptions containing negations.

## Steps

**Gathering the images**

1. Unzip MSCOCO.zip, or alternatively download the training and val JSON files to a folder called MSCOCO in this directory.
2. Run `python coco_img_urls.py`. This generates `coco_image_links.txt`.
3. Run `bash download.sh` to download all the images and move them to the images folder.

**Generating the image features**

1. Open `build_dataset.sh` and change the path to the image folder to the one on your system. (I used an absolute path.)
2. Download the VGG ILSVRC 16-layer model [from here](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) to the `model` folder.
3. Run `bash build_dataset.sh`. This takes care of the arguments for `standalone.py`, which is a script to extract image features and generate a .h5 dataset file, with a matching JSON file.
