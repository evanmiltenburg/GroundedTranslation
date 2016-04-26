import json
import itertools
from collections import defaultdict

with open('MSCOCO/captions_train2014.json') as f1,\
     open('MSCOCO/captions_val2014.json') as f2:
     d1 = json.load(f1)
     d2 = json.load(f2)

# Negations: match the word.
# ADJECTIVES = {"absent", "away", "clear", "deprived", "devoid", "free", "removed", "stripped", "vanished"}
# ADVERBS = {"barely", "hardly", "scarcely"}
FREE_NEG = {"not", "n't"}
NO_NEG = {"never", "no", "none", "nothing", "nobody", "nowhere", "nor", "neither"}
PREPOSITIONS = {"without", "sans", "minus"}#, "except", "from", "out", "off"}
# NPIS = {"any","anything"} # to make sure we don't miss anything.

# with open('./negations.csv') as f:
#     reader = csv.reader(f)
#     AFFIXED = {word for word, yes_no in reader if yes_no == 'yes'}

# Negations: special cases.
# PREFIXES = {"a", "dis", "in", "im", "non", "un"}
# SUFFIXES = {"less"}
VERBS = {"lack", "omit", "miss", "fail"}
# All.
TO_MATCH = FREE_NEG | NO_NEG | PREPOSITIONS

def contains_negation(doc):
    "Checks whether the line contains a negation."
    tokenized = doc.split()
    bag_of_words = set(tokenized)
    matches = TO_MATCH & bag_of_words
    if matches:
        return True
    for word in bag_of_words:
        for verb in VERBS:
            if word.startswith(verb):
                return True
    return False

def get_image_links(d):
    "Get a set of image links that contain a negation."
    ids = {entry['image_id'] for entry in d1['annotations']
                             if contains_negation(entry['caption'])}
    return {image['coco_url'] for image in d['images'] if image['id'] in ids}

# Get image links
image_links = get_image_links(d1) | get_image_links(d2)

# Write them to a file.
with open('coco_image_links.txt','w') as f:
    f.writelines(link + '\n' for link in image_links)
