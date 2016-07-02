import h5py

# Older files:
standard = h5py.File('flickr30k/dataset.h5', 'r')
undersampled = h5py.File('flickr30k/undersampled.h5', 'r')
coco_selected = h5py.File('coco/selected_dataset.h5', 'r')
coco_undersampled = h5py.File('coco/undersampled_dataset.h5', 'r')

# New files
extended = h5py.File('extended.h5')
maximal = h5py.File('maximal.h5')

# Add basic Flickr30K data.
standard.copy('train', extended)
standard.copy('val', extended)
standard.copy('test', extended)

# Add basic Flickr30K data.
undersampled.copy('train', maximal)
undersampled.copy('val', maximal)
undersampled.copy('test', maximal)

# Add the coco data.
for key in coco_selected['train']:
    group = extended['train'].create_group(key)
    text_data = group.create_dataset("descriptions", (5,), dtype=h5py.special_dtype(vlen=unicode))
    image_data = group.create_dataset("img_feats", (4096,), dtype='float32')
    text_data[:] = coco_selected['train'][key]['descriptions']
    image_data[:] = coco_selected['train'][key]['img_feats']

# Add the coco data.
for key in coco_undersampled['train']:
    group = maximal['train'].create_group(key)
    text_data = group.create_dataset("descriptions", (1,), dtype=h5py.special_dtype(vlen=unicode))
    image_data = group.create_dataset("img_feats", (4096,), dtype='float32')
    text_data[:] = coco_undersampled['train'][key]['descriptions']
    image_data[:] = coco_undersampled['train'][key]['img_feats']


print 'Length of standard:', len(standard['train'])
print 'Length of coco selected:', len(coco_selected['train'])
print 'Combined:', len(standard['train']) + len(coco_selected['train'])
print 'Length of extended:', len(extended['train'])

print '----------------------------'

print 'Length of undersampled:', len(undersampled['train'])
print 'Length of coco undersampled:', len(coco_undersampled['train'])
print 'Combined:', len(undersampled['train']) + len(coco_undersampled['train'])
print 'Length of maximal:', len(maximal['train'])
