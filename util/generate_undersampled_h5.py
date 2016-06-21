"""
Put this in the same directory as your Flickr30K dataset.h5 file, along with a
copy of the annotations from Van Miltenburg et al. (2016):
https://github.com/evanmiltenburg/annotating-negations/blob/master/results/final_annotations.tsv

Be sure to make a folder called 'undersampled'.
"""

import h5py
import csv
import random

random.seed(123456789)

with open('final_annotations.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    negation_sentences = set()
    not_a_description = set()
    ignore = set()
    for entry in reader:
        sentence = entry['Sentence'].strip(' .')
        if entry['Category'] == 'Not a description/Meta':
            not_a_description.add(sentence)
        elif entry['Category'] == 'False positive':
            ignore.add(sentence)
        else:
            negation_sentences.add(sentence)

old_file = h5py.File('dataset.h5','r')
new_file = h5py.File('undersampled/dataset.h5','a')

def fill_new_h5file(old_file, new_file, split):
    "Function to fill the new h5 file with only one description per image."
    # Ensure the function is called with the correct args.
    assert split in {'train','val'}
    
    # Create group for the current split.
    current_split = new_file.create_group(split)
    
    # Fill the new file.
    for key in old_file[split]:
        # Get a description for the current key.
        descriptions = set(old_file[split][key]['descriptions']) - not_a_description
        intersection = descriptions & negation_sentences
        if len(intersection) > 0:
            sentence_to_use = random.choice(list(intersection))
        else:
            sentence_to_use = random.choice(list(descriptions))
        
        # Create group & datasets for current key.
        key_group = current_split.create_group(key)
        text_data = key_group.create_dataset("descriptions", (1,), dtype=h5py.special_dtype(vlen=unicode))
        image_data = key_group.create_dataset("img_feats", (4096,), dtype='float32')
        
        # Fill datasets.
        text_data[0] = sentence_to_use
        image_data[:] = old_file[split][key]['img_feats']
    
    # Flush to disk.
    new_file.flush()

fill_new_h5file(old_file, new_file, 'train')
fill_new_h5file(old_file, new_file, 'val')

old_file.close()
new_file.close()
