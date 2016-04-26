import sys
import os.path
import argparse
import glob
import json
import h5py
import numpy as np
from scipy.misc import imread, imresize

parser = argparse.ArgumentParser("Extracts features and generates a h5 file.")
parser.add_argument('--caffe',
                    help='path to caffe installation')
parser.add_argument('--model_def',
                    help='path to model definition prototxt')
parser.add_argument('--model',
                    help='path to model parameters')
parser.add_argument('--files',
                    help='path to a file containing a list of images')
parser.add_argument('--gpu',
                    action='store_true',
                    help='whether to use gpu training')
parser.add_argument('--name',
                    help='name of the new h5 dataset')

args = parser.parse_args()

if args.name:
    h5filename = args.name
else:
    h5filename = "customimages.h5"
    
if os.path.exists(h5filename):
    print("dataset.h5 already exists at that path. Refusing to overwrite")
    sys.exit(1)

if args.caffe:
    caffepath = args.caffe + '/python'
    sys.path.append(caffepath)

import caffe

def predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    features = out[net.outputs[0]]
    return features


def batch_predict(filenames, net):
    """
    Get the features for all images from filenames using a network

    Inputs:
    filenames: a list of names of image files

    Returns:
    an array of feature vectors for the images in that file
    """

    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    F = net.blobs[net.outputs[0]].data.shape[1]
    Nf = len(filenames)
    Hi, Wi, _ = imread(filenames[0]).shape
    allftrs = np.zeros((Nf, F))
    for i in range(0, Nf, N):
        in_data = np.zeros((N, C, H, W), dtype=np.float32)

        batch_range = range(i, min(i+N, Nf))
        batch_filenames = [filenames[j] for j in batch_range]
        Nb = len(batch_range)

        batch_images = np.zeros((Nb, 3, H, W))
        for j,fname in enumerate(batch_filenames):
            im = imread(fname)
            if len(im.shape) == 2:
                im = np.tile(im[:,:,np.newaxis], (1,1,3))
            # RGB -> BGR
            im = im[:,:,(2,1,0)]
            # mean subtraction
            im = im - np.array([103.939, 116.779, 123.68])
            # resize
            im = imresize(im, (H, W), 'bicubic')
            # get channel in correct dimension
            im = np.transpose(im, (2, 0, 1))
            batch_images[j,:,:,:] = im

        # insert into correct place
        in_data[0:len(batch_range), :, :, :] = batch_images

        # predict features
        ftrs = predict(in_data, net)

        for j in range(len(batch_range)):
            allftrs[i+j,:] = ftrs[j,:]

        print 'Done %d/%d files' % (i+len(batch_range), len(filenames))

    return allftrs

if args.gpu:
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()

net = caffe.Net(args.model_def, args.model, caffe.TEST)

filenames = []

base_dir = os.path.dirname(args.files)
with open(args.files) as fp:
    for line in fp:
        filename = os.path.join(base_dir, line.strip().split()[0])
        filenames.append(filename)

features_struct = batch_predict(filenames, net)

def sentences(i):
    "Returns placeholder sentences."
    placeholder = "a researcher is writing an improbably long sentence just for the sake of avoiding an error because that would happen if this silly sentence were just a bit shorter"
    tokenized = placeholder.split()
    return [{"tokens":tokenized, "raw":placeholder +" .", "imgid":i, "sentid": i}]

def images(filenames):
    "Returns the image dictionary for a given path."
    for i, filename in enumerate(filenames):
        yield {"filename":filename,
               "sentids":[i],
               "imgid":i,
               "split":"val",
               "sentences": sentences(i)}

def json_dictionary(filenames, name):
    "Generates a dictionary similar to the JSON format."
    return {"images":list(images(filenames)),
            "dataset":name}

# Get JSON data.
jdata = json_dictionary(filenames, h5filename)

# Write it to a file.
with open(h5filename + ".json",'w') as f:
    json.dump(jdata,f)

# Open h5 file and write the contents to that file as well.
h5output = h5py.File(h5filename, "w")
val = h5output.create_group("val")

for idx, image in enumerate(jdata['images']):
    image_filename = image['filename']
    image_id = image['imgid']
    container = val.create_group("%06d" % idx)
    # The descriptions "Dataset" contains one row per description in unicode
    text_data = container.create_dataset("descriptions", (len(image['sentences']),),
                                         dtype=h5py.special_dtype(vlen=unicode))
    # The visual features "Dataset" contains one vector array per image in float32
    # (Changed from per description: made dataset very large and redudant)
    image_data = container.create_dataset("img_feats",
                                        (4096,), dtype='float32')
                                        # (NUM_DESCRIPTIONS, 4096), dtype='float32')
    image_data[:] = features_struct[idx][:]
    for idx2, text in enumerate(image['sentences']):
        text_data[idx2] = " ".join(text['tokens']) #['raw']
        #image_data[idx2] = features_struct[:,idx]

    if idx % 100 == 0:
        print("Processed %d/%d instances" % (idx, len(jdata['images'])))

# Create empty groups:
h5output.create_group("test")

# Create placeholder description.
train = h5output.create_group("train")
container = train.create_group('999999')
text_data = container.create_dataset("descriptions", (1,), dtype=h5py.special_dtype(vlen=unicode))
text_data[0] = "a researcher is writing an improbably long sentence just for the sake of avoiding an error because that would happen if this silly sentence were just a bit shorter"

h5output.close()
