# Change the directory if the folder is somewhere else.
ls -d -1 /data/COCO/images/*.jpg > list_of_images.txt

# Create the dataset.
python standalone.py --name=dataset.h5 --files=list_of_images.txt --model=model/VGG_ILSVRC_16_layers.caffemodel --model_def=model/deploy_features.prototxt

# Create mapping between dataset IDs and file IDs
python mapping.py
