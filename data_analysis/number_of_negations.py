import json
import glob
from collections import Counter

import matplotlib.pyplot as plt
import seaborn

plt.switch_backend('agg')

FREE_NEG = {"not", "n't"}
NO_NEG = {"never", "no", "none", "nothing", "nobody", "nowhere", "nor", "neither"}
PREPOSITIONS = {"without", "sans", "minus"}#, "except", "from", "out", "off"}
# All.
TO_MATCH = FREE_NEG | NO_NEG | PREPOSITIONS
VERBS = {"lack", "omit", "miss", "fail"}

gen = glob.glob('../JSON/flickr30k/gen*.json')
mbs = glob.glob('../JSON/flickr30k/MBS*.json')
coco = glob.glob('../JSON/coco/*.json')

def analyze_json(path):
    'Analyze JSON file for negations'
    counter = Counter()
    with open(path) as f:
        d = json.load(f)
        total = len(d)
    for line in d.values():
        tokenized = line.split()
        bag_of_words = set(tokenized)
        counter.update(TO_MATCH & bag_of_words)
        for word in bag_of_words:
            for verb in VERBS:
                if word.startswith(verb):
                    counter.update(verb)
    return counter, total

def beam_size_from_path(path):
    "Get beam size from path"
    return int(path[:-5].split('_')[-1])

def percentage_negations(path):
    "Return percentage of negations for a given json file."
    c, total = analyze_json(path)
    num_negations = sum(c.values())
    pct = (float(num_negations)/total) * 100
    print("path: "+ path)
    print("total number of negations: " + str(num_negations))
    print("percentage: " + str(pct))
    return pct

def get_values(paths):
    "Get values to plot. X=beam size, Y=amount of negations"
    coords = [(beam_size_from_path(path),percentage_negations(path) ) for path in paths]
    x,y = zip(*sorted(coords))
    return x,y

gen_x, gen_y = get_values(gen)
mbs_x, mbs_y = get_values(mbs)
coco_x, coco_y = get_values(coco)

gen_line, = plt.plot(gen_x, gen_y, '-o', label='Beam search')
mbs_line, = plt.plot(mbs_x, mbs_y, '-o', label='Dual beam search')
coco_line, = plt.plot(coco_x, coco_y, '-o', label='Dual beam search (COCO)')

plt.legend(handles=[gen_line, mbs_line, coco_line],loc=2)
plt.xlabel('Beam size')
plt.ylabel('% negations')

plt.savefig('number_of_negations.pdf')

# Let's make another figure.
plt.clf()

def number_of_kinds(path):
    "Get number of kinds."
    c,total = analyze_json(path)
    return len(c)

def get_kinds_values(paths):
    "Get values to plot. X=beam size, Y=amount of negations"
    coords = [(beam_size_from_path(path),number_of_kinds(path)) for path in paths]
    x,y = zip(*sorted(coords))
    return x,y

mbs_x, mbs_y = get_kinds_values(mbs)
coco_x, coco_y = get_kinds_values(coco)

mbs_line, = plt.plot(mbs_x, mbs_y, '-o', label='Flickr30K')
coco_line, = plt.plot(coco_x, coco_y, '-o', label='COCO')

plt.legend(handles=[mbs_line, coco_line],loc=2)

plt.xlabel('Beam size')
plt.ylabel('Kinds of negations')

plt.savefig('kinds_of_negations.pdf')
