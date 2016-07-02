"""
Put this in the same directory as your Flickr30K dataset.h5 file, along with a
copy of the annotations from Van Miltenburg et al. (2016):
https://github.com/evanmiltenburg/annotating-negations/blob/master/results/final_annotations.tsv

Be sure to make a folder called 'undersampled'.
"""

import h5py
import csv
import random
import string
from collections import Counter

ASCII = set(string.ascii_lowercase)
random.seed(123456789)

def sentence_hash(sentence):
    "Sentence identity check."
    # apparently the ampersands are kept in this version of the Flickr30K corpus,
    # but the HTML conversion also introduced these silly codes.
    sentence = sentence.replace(' & ', ' &amp ')
    return ''.join([char for char in sentence.lower().encode('ascii','ignore')
                         if char in ASCII])
    # c = Counter(char for char in sentence.lower() if char in ASCII)
    # return ''.join([str(x) for pair in sorted(c.items()) for x in pair])

def hash_dict(list_of_sentences):
    "Create dictionary of sentence hashes."
    return {sentence_hash(s): s for s in list_of_sentences}

with open('final_annotations.tsv') as f:
    reader = csv.DictReader(f, delimiter='\t')
    negation_sentence_hashes = dict()
    not_a_description = set()
    ignore = set()
    for entry in reader:
        h = sentence_hash(entry['Sentence'])
        if entry['Category'] == 'Not a description/Meta':
            not_a_description.add(h)
        elif entry['Category'] == 'False positive':
            ignore.add(h)
        else:
            negation_sentence_hashes[h] = entry['Sentence'].lower()

negation_sentences = set(negation_sentence_hashes.keys())

print "This should be empty:", not_a_description & negation_sentences

print len(negation_sentences), "examples in the set of negation sentences."
old_file = h5py.File('flickr30k/dataset.h5','r')
new_file = h5py.File('flickr30k/undersampled.h5','a')

found = set()

def fill_new_h5file(old_file, new_file, split):
    "Function to fill the new h5 file with only one description per image."
    # Ensure the function is called with the correct args.
    assert split == 'train'

    # Create group for the current split.
    current_split = new_file.create_group(split)

    # Negation counter
    neg_counter = {'all_negs': 0, 'single_neg': 0}
    # Fill the new file.
    for key in old_file[split]:
        # Get a description for the current key.
        hd = hash_dict(old_file[split][key]['descriptions'])
        descriptions = set(hd.keys()) - not_a_description
        intersection = descriptions & negation_sentences
        neg_counter['all_negs'] += len(intersection)
        global found
        found.update(intersection)
        if len(intersection) > 0:
            matches = [s for h,s in hd.items() if h in intersection]
            sentence_to_use = random.choice(matches)
            neg_counter['single_neg'] += 1
        else:
            descriptions = list(hd.values())
            sentence_to_use = random.choice(descriptions)

        # Create group & datasets for current key.
        key_group = current_split.create_group(key)
        text_data = key_group.create_dataset("descriptions", (1,), dtype=h5py.special_dtype(vlen=unicode))
        image_data = key_group.create_dataset("img_feats", (4096,), dtype='float32')

        # Fill datasets.
        text_data[0] = sentence_to_use
        image_data[:] = old_file[split][key]['img_feats']

    # Flush to disk.
    new_file.flush()

    # Report stats
    return neg_counter

negs = fill_new_h5file(old_file, new_file, 'train')
print(negs)
print "For", str(len(new_file['train'])), "images."
print "Percentage (full): ", (negs['all_negs']/float(len(new_file['train']) * 5)) * 100
print "Percentage (conc): ", (negs['single_neg']/float(len(new_file['train']))) * 100


# And fill the rest of the file.
old_file.copy('/val', new_file)
old_file.copy('/test', new_file)

def optional_check():
    "Optionally check for sentences that we failed to match."
    print 'Not found:'
    not_found = {s for h,s in negation_sentence_hashes.items() if not h in found}
    print '\n'.join(not_found)

    from Levenshtein import distance

    print ''
    print 'Computing Levenshtein distances to find candidates we could have missed.'

    found_missing = False
    for split in ['train']:
        for key in old_file[split]:
            for description in old_file[split][key]['descriptions']:
                for sentence in not_found:
                    h1 = sentence_hash(sentence)
                    h2 = sentence_hash(description)
                    if distance(h1, h2) < 15:
                        found_missing = True
                        print 'POSSIBLE MATCH:'
                        print sentence
                        print description
                        print '----------------------------------'

    if not found_missing:
        print 'Matched all we could possibly match.'

old_file.close()
new_file.close()
