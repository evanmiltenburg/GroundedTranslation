# Download the data
wget http://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
wget http://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip

# Unzip the data
unzip flickr30k.zip
unzip coco.zip

# Delete the zip files to save space
rm flickr30k.zip
rm coco.zip

# create h5 files
python ../jsonmat2h5.py --path flickr30k
python ../jsonmat2h5.py --path coco

# generate the relevant test files
python generate_undersampled_h5.py
python select_coco_negations.py
python generate_coco_h5.py
python merge_coco_flickr.py

# make dirs
mkdir standard
mkdir condensed
mkdir extended
mkdir maximal

# order files
mv flickr30k/dataset.h5 standard/dataset.h5
mv flickr30k/undersampled.h5 condensed/dataset.h5
mv extended.h5 extended/dataset.h5
mv maximal.h5 maximal/dataset.h5
