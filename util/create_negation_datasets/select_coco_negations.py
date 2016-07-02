from collections import defaultdict, namedtuple
import csv
import h5py

# Negations: match the word.
FREE_NEG = {"not", "n't"}
NORMALIZED = {"dont", "werent", "isnt", "doesnt", "couldnt", "shouldnt", "hasnt",
              "wont", "wouldnt", "cant", "arent", "havent", "didnt"}
NO_NEG = {"never", "no", "none", "nothing", "nobody", "nowhere", "nor", "neither"}
PREPOSITIONS = {"without", "sans", "minus"}#, "except", "from", "out", "off"}

# All.
TO_MATCH = FREE_NEG | NO_NEG | PREPOSITIONS | NORMALIZED
VERBS = {"lack", "omit", "miss", "fail"}

# Heuristics for throwing out descriptions. Recall that all descriptions here contain
# negations, which are also a good filter by themselves. Of course all flagged results
# need to be manually checked.
BAD_DESCRIPTION_HEURISTICS = {'picture','image','photo','see','understand'}

def description_generator():
    "Generate all descriptions for a particular dataset."
    h5file = h5py.File('coco/dataset.h5', 'r')
    for split in ['test', 'train', 'val']:
        for key in h5file[split]:
            for description in h5file[split][key]['descriptions']:
                yield [split, key, description]

Row = namedtuple('Row',['split', 'key', 'match', 'keep', 'description','heuristic'])

def match_generator():
    "Generate negation matches."
    for split, key, description in description_generator():
        tokenized = description.split()
        bag_of_words = set(tokenized)
        overlap = TO_MATCH & bag_of_words
        if overlap:
            yield Row(split, key, '|'.join(overlap), 'keep', description, bool(BAD_DESCRIPTION_HEURISTICS & bag_of_words))
        for word in bag_of_words:
            for verb in VERBS:
                if word.startswith(verb):
                    yield Row(split, key, verb, 'keep', description, bool(BAD_DESCRIPTION_HEURISTICS & bag_of_words))

gen = match_generator()
sorted_rows = sorted(gen, key=lambda row:(row.heuristic,row.description), reverse=True)

with open('coco_negations.tsv','w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(sorted_rows)
