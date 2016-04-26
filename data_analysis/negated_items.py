import glob
import json
from collections import Counter

mbs = glob.glob('../JSON/flickr30k/MBS*.json')
coco = glob.glob('../JSON/coco/*.json')

def analyze_json(path):
    items = []
    with open(path) as f:
        d = json.load(f)
    for line in d.values():
        tokenized = line.split()
        if 'no' in tokenized:
            no_index = tokenized.index('no')
            try:
                item = tokenized[no_index + 1]
            except IndexError:
                item = None
            items.append(item)
                
        elif 'without' in tokenized:
            without_index = tokenized.index('without')
            if without_index == (len(tokenized)-1):
                item = None
            else:
                try:
                    w1, w2 = tokenized[without_index + 1 : without_index + 3]
                    if w1 in ['a','the','any']:
                        item = w2
                    else:
                        item = w1
                except IndexError:
                    item = tokenized[without_index+1]
            items.append(item)
    return Counter(items)

def analyze_all(file_list):
    counter = Counter()
    for path in file_list:
        new_counter = analyze_json(path)
        counter += new_counter
    return counter


mbs_counter = analyze_all(mbs)
coco_counter = analyze_all(coco)

print('mbs:')
print(mbs_counter)
print('coco')
print(coco_counter)
