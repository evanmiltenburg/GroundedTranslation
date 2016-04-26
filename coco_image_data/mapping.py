import json

with open('dataset.h5.json') as f:
    d = json.load(f)

def get_file_id(image):
    "Get the file id."
    return image['filename'].split('/')[-1][:-4]

with open('mapping.json','w') as f:
    mapping = {image['imgid']: get_file_id(image) for image in d['images']}
    json.dump(mapping, f)
