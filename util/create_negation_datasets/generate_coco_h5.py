import csv
import h5py
from collections import defaultdict
import random

# Make this reproducible.
random.seed(123456789)

# Load original data.
source_file = h5py.File('coco/dataset.h5', 'r')

def zfill(s):
    "Add zeroes untill string width is 6. Defined here for compatibility with python 2/3."
    zeroes = 6 - len(s)
    return ('0' * zeroes) + s


def get_sents():
    "Create dictionary with the pictures to keep, and a list of descriptions containing negations."
    blacklist = set()
    sents = defaultdict(list)
    
    with open('coco_negations_edited.tsv') as f:
        fieldnames = ['split', 'key', 'match', 'keep', 'description','heuristic']
        reader = csv.DictReader(f,fieldnames=fieldnames, delimiter='\t')
        for row in reader:
            key = (row['split'], zfill(row['key']))
            description = row['description']
            # Either keep the description...
            if row['keep'] == 'keep':
                sents[key].append(description)
            # Or add the split and ID combination to the blacklist.
            else:
                blacklist.add(key)

    # Remove all the pictures that have a bad description.
    for key in blacklist:
        if key in sents:
            del sents[key]

    return sents

def build_datasets(sents):
    # Make files:
    selected_coco = h5py.File('coco/selected_dataset.h5', 'w')
    undersampled_coco = h5py.File('coco/undersampled_dataset.h5', 'w')
    
    # Make groups:
    selected_train = selected_coco.create_group('train')
    undersampled_train = undersampled_coco.create_group('train')
    for split, key in sents:
        imgid = '_'.join(['coco', split, key])
        sents_with_negations = sents[(split,key)]
        description = random.choice(sents_with_negations)
        
        # Create ID group.
        selected_id = selected_train.create_group(imgid)
        # Use 5 sentences.
        text_data = selected_id.create_dataset("descriptions", (5,), dtype=h5py.special_dtype(vlen=unicode))
        image_data = selected_id.create_dataset("img_feats", (4096,), dtype='float32')
        try:
            text_data[:] = source_file[split][key]['descriptions']
        except TypeError:
            # If the number of descriptions is greater than 5..
            descriptions = set(source_file[split][key]['descriptions']) - {description}
            selected_descriptions = [description] + random.sample(descriptions,4)
            text_data[:] = selected_descriptions
        image_data[:] = source_file[split][key]['img_feats']
        
        # Create ID group.
        undersampled_id = undersampled_train.create_group(imgid)
        # Use 1 sentence.
        text_data = undersampled_id.create_dataset("descriptions", (1,), dtype=h5py.special_dtype(vlen=unicode))
        image_data = undersampled_id.create_dataset("img_feats", (4096,), dtype='float32')
        text_data[0] = description
        image_data[:] = source_file[split][key]['img_feats']
    
    # Flush to disk.
    selected_coco.flush()
    undersampled_coco.flush()

# Execute:
sents = get_sents()

print "number of images:", len(sents)
print "number of negations:", sum(len(descriptions) for descriptions in sents.values())

build_datasets(sents)
