"""
Data processing for VisualWordLSTM happens here; this creates a class that
acts as a data generator/feed for model training.
"""
from __future__ import print_function

from collections import defaultdict
import cPickle
import h5py
import logging
import numpy as np
np.set_printoptions(threshold='nan')
import os
import sys
import random

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Strings for beginning, end of sentence, padding
# These get specified indices in word2index
BOS = "<S>"  # index 1
EOS = "<E>"  # index 2
PAD = "<P>"  # index 0

# Dimensionality of image feature vector
IMG_FEATS = 4096

# How many descriptions to use for training if "--small" is set.
SMALL_NUM_DESCRIPTIONS = 300
SMALL_VAL = 100


class VisualWordDataGenerator(object):
    """
    Creates input arrays for VisualWordLSTM and deals with input dataset in
    general. Input dataset must now be in HTF5 format.

    Important methods:
        yield_training_batch() is a generator function that yields large
            batches from the training data split (in case training is too
            large to fit in memory)
        get_data_by_split(split) returns the required arrays (descriptions,
            images, targets) for the dataset split (train/val/test/as given by
            the hdf5 dataset keys)
    """
    def __init__(self, args_dict, input_dataset=None, hsn=0):
        """
        Initialise data generator: this involves loading the dataset and
        generating vocabulary sizes.
        If dataset is not given, use flickr8k.h5.

        hsn is now officially unused. Will be removed after I deal with
        extract_hidden_features.py
        """
        logger.info("Initialising data generator")
        self.args_dict = args_dict

        # size of chucks that the generator should return;
        # if 0 returns full dataset at once.
        # self.big_batch_size = args_dict.big_batch_size

        # Number of descriptions to return per image.
        self.num_sents = args_dict.num_sents  # default 5 (for flickr8k)
        self.unk = args_dict.unk  # default 5

        self.small = args_dict.small  # default False
        if self.small:
            logger.warn("--small: Truncating datasets!")
        self.run_string = args_dict.run_string

        # self.datasets holds 1+ datasets, where additional datasets will
        # be used for supertraining the model
        self.datasets = []
        self.openmode = "r+" if self.args_dict.h5_writeable else "r"
        if not input_dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", self.openmode)
        else:
            self.dataset = h5py.File("%s/dataset.h5" % input_dataset, self.openmode)
        logger.info("Train/val dataset: %s", input_dataset)

        if args_dict.supertrain_datasets is not None:
            for path in args_dict.supertrain_datasets:
                logger.info("Adding supertrain datasets: %s", path)
                self.datasets.append(h5py.File("%s/dataset.h5" % path, "r"))
        self.datasets.append(self.dataset)

        # hsn doesn't have to be a class variable.
        # what happens if self.hsn is false but hsn_size is not zero?
        if self.args_dict.source_vectors is not None:
            self.source_dataset = h5py.File("%s/dataset.h5"
                                            % self.args_dict.source_vectors,
                                            "r")
            self.source_encoder = args_dict.source_enc
            self.source_type = args_dict.source_type
            self.source_dim = args_dict.hidden_size
            self.h5_dataset_str = "%s-hidden_feats-%s-%d" % (self.source_type,
                                                      self.source_encoder,
                                                      self.source_dim)
            self.hsn_size = len(self.source_dataset['train']['000000']
                                [self.h5_dataset_str])
            logger.info("Reading source vectors from %s with %d dims",
                        self.h5_dataset_str, self.hsn_size)

        # These variables are filled by extract_vocabulary
        self.word2index = dict()
        self.index2word = dict()
        # This is set to include BOS & EOS padding
        self.max_seq_len = 0
        # Can check after extract_vocabulary what the actual max seq length is
        # (including padding)
        self.actual_max_seq_len = 0

        # This counts number of descriptions per split
        # Ignores test for now (change in extract_vocabulary)
        self.split_sizes = {'train': 0, 'val': 0, 'test': 0}
        self.calculate_split_sizes()

        # These are used to speed up the validation process
        self._cached_val_input = None
        self._cached_val_targets = None
        self._cached_references = None

        if self.args_dict.use_predicted_tokens and self.args_dict.no_image:
          logger.info("Input predicted descriptions")
          self.ds_type = 'predicted_description'
        else:
          logger.info("Input gold descriptions")
          self.ds_type = 'descriptions'

    def get_vocab_size(self):
        """Return training (currently also +val) vocabulary size."""
        return len(self.word2index)

    def get_new_training_arrays(self, array_size, use_sourcelang, use_image):
        """ Get empty arrays for yield_training_batch. """
        arrays = []
        # dscrp_array at arrays[0]
        arrays.append(np.zeros((array_size,
                                self.max_seq_len,
                                len(self.word2index))))
        if use_sourcelang:  # hsn_array at arrays[1] (if used)
            arrays.append(np.zeros((array_size,
                                    self.max_seq_len,
                                    self.hsn_size)))
        if use_image:  # at arrays[2] or arrays[1]
            arrays.append(np.zeros((array_size,
                                    self.max_seq_len,
                                    IMG_FEATS)))
        return arrays

    def yield_training_batch(self, big_batch_size, use_sourcelang=False,
                             use_image=True, return_keys=False):
        """
        Returns a batch of training examples.

        Uses hdf5 dataset.
        """
        assert big_batch_size > 0
        logger.info("Generating training data batch")

        arrays = self.get_new_training_arrays(big_batch_size,
                                              use_sourcelang, use_image)
        num_descriptions = 0  # total num descriptions found so far.
        batch_max_seq_len = 0
        discarded = 0  # total num of discards
        # KEYS ARE OVER IMAGES NOT DESCRIPTIONS
        # THIS WILL BREAK IF THERE ARE MULTIPLE DESCRIPTIONS/IMAGE
        keys = []
        if use_sourcelang and use_image:  # where is image array in arrays.
            img_idx = 2
        else:
            img_idx = 1
        # Iterate over *images* in training splits
        for dataset in self.datasets:
            #training_size = self.split_sizes['train'] # this won't work with multiple datasets
            training_size = self.split_sizes['train']
            logging.info("training size %d", training_size)
            if self.small:
                training_size = SMALL_NUM_DESCRIPTIONS

            for data_key in dataset['train']:
                ds = dataset['train'][data_key][self.ds_type][0:self.args_dict.num_sents]
                for description in ds:
                    batch_index = num_descriptions % big_batch_size
                    # Return (filled) big_batch array
                    if (batch_index == 0) and (num_descriptions > 0):
                        # Truncate descriptions to max length of batch (plus
                        # 3, for padding and safety)
                        for i, arr in enumerate(arrays):
                            arrays[i] = arr[:, :(batch_max_seq_len + 3), :]
                        targets = self.get_target_descriptions(arrays[0])

                        if return_keys:
                            yield (arrays, targets, num_descriptions ==
                                   training_size, keys)
                        else:
                            yield (arrays, targets, num_descriptions == training_size)

                        # Testing multiple big batches
                        if self.small and num_descriptions >= training_size:
                            logger.warn("Breaking out of yield_training_batch")
                            raise StopIteration

                        # Create new training arrays that are either
                        # big_batch_size or the size of the number of
                        # descriptions left
                        this_bbs = big_batch_size
                        if (training_size - num_descriptions) < big_batch_size:
                            this_bbs = training_size - num_descriptions - discarded
                            logger.info("Creating (truncated) final big batch, size %d", this_bbs)
                        arrays = self.get_new_training_arrays(this_bbs,
                                                              use_sourcelang, use_image)
                        keys = []

                    if len(description.split()) > batch_max_seq_len:
                        batch_max_seq_len = len(description.split())

                    try:
                        arrays[0][batch_index, :, :] = self.format_sequence(
                            description.split())
                        if use_sourcelang:
                            arrays[1][batch_index, 0, :] =\
                                self.get_source_features('train', data_key)
                                #self.get_hsn_features('train', data_key)
                        if use_image:
                            # img_idx is 1 or 2, depending on use_sourcelang
                            arrays[img_idx][batch_index, 0, :] =\
                                self.get_image_features(dataset, 'train', data_key)
                        num_descriptions += 1
                        keys.append(data_key)
                    except AssertionError:
                        # thrown by format_sequence() when we cannot encode
                        # any words in the sentence
                        discarded += 1
                        continue
                    except KeyError:
                        # thrown by get_hsn_features() when we cannot retrieve
                        # a source-language feature vector
                        discarded += 1
                        continue

            # End of looping through a dataset (may be one of many).
            # Yield final batch for this dataset.
            # batch_index is not guaranteed to modulo zero on the
            # final big_batch. This if statement catches that and
            # resizes the final yielded dscrp/img_array and targets
            batch_index = num_descriptions % big_batch_size
#           if batch_index != 0:
#               logging.warn("resizing final batch", batch_index,
#                            num_descriptions)
#               arrays = self.resize_arrays(batch_index, arrays)
            #assert batch_index == arrays[0].shape[0],\
            #        "Filled array %d, size %d" % (batch_index, arrays[0].shape[0])
            assert num_descriptions == training_size - discarded,\
                    "Yielded %d descriptions, train size %d (%d)" % (
                        num_descriptions,  training_size, self.split_sizes['train'])

            # Return (filled) big_batch array
            # Truncate descriptions to max length of batch (plus
            # 3, for padding and safety)
            for i, arr in enumerate(arrays):
                arrays[i] = arr[:, :(batch_max_seq_len + 3), :]
            targets = self.get_target_descriptions(arrays[0])
            # this will do validation at the end of every training dataset (in supertraining)
            if return_keys:
                yield (arrays, targets, True, keys)
            else:
                yield (arrays, targets, True)
            # For next dataset (i.e. if supertraining)
            arrays = self.get_new_training_arrays(
                big_batch_size, use_sourcelang, use_image)
            discarded = 0
            keys = []

    def resize_arrays(self, new_size, arrays):
        """
        Resize all the arrays to new_size along dimension 0.
        Sometimes we need to initialise a np.zeros() to an arbitrary size
        and then cut it down to out intended new_size.
        Now used only with small_val.
        """
        logger.debug("Resizing batch_size in structures from %d -> %d",
                    arrays[0].shape[0], new_size)

        for i, array in enumerate(arrays):
            arrays[i] = np.resize(array, (new_size, array.shape[1],
                                          array.shape[2]))
        return arrays

    def get_refs_by_split_as_list(self, split):
        """ Replaces extract_references in Callbacks."""

        #if self._cached_references is not None and split == "val":
        #    logger.info("Retrieving cached val_references")
        #    return self._cached_references

        # Doesn't work for train because of size/batching, also not needed.
        assert split in ['test', 'val'], "Not possible for split %s" % split
        references = []
        for data_key in self.dataset[split]:
            this_image = []
            for descr in self.dataset[split][data_key]['descriptions']:
                this_image.append(descr)
            references.append(this_image)
            if self.args_dict.small_val and split == 'val':
                if len(references) >= SMALL_VAL:
                    break

        #if split == "val":
        #    self._cached_references = references
        return references

    def get_generation_data_by_split(self, split, use_sourcelang=False,
                                     use_image=True):
        """ Gets all input data for model for a given split (ie. train, val,
        test).
        Returns tuple containing a list of training input arrays (depending on
        use_sourcelang and use_image) and a target array.
        Training arrays may contain (in this order)
            descriptions: input array for text LSTM [item, timestep,
                vocab_onehot] (necessary)
            source language:  input array of hidden state vectors for source
                language features (optional)
            image: input array of image features [item, timestep, img_feats at
                    timestep=0 else 0] (optional)
        Targets: target array for text LSTM (same format and data as
                descriptions, timeshifted)

        Changed: If small_val is set, the original arrays are now also a bit
        smaller than split_size (and then truncated to d_idx, as before). This
        is so I can run this on a lower-memory machine without thrashing.
        """

        logger.info("Making generation data for %s", split)
        if len(self.datasets) > 1 and split == "train":
            logger.warn("Called get_data_by_split on train while supertraining;\
                        this is probably NOT WHAT YOU INTENDED")

        split_size = self.split_sizes[split]
        intended_size = np.inf
        if self.args_dict.small_val:
            intended_size = SMALL_VAL
            # Make split_size comfortably bigger than intended_size
            split_size = 2 * SMALL_VAL * self.args_dict.num_sents

        arrays = self.get_new_training_arrays(split_size, use_sourcelang,
                                              use_image)

        if use_sourcelang and use_image:  # where is image array in arrays.
            img_idx = 2
        else:
            img_idx = 1

        d_idx = 0  # description index
        for data_key in self.dataset[split]:
            descriptions = self.dataset[split][data_key]['descriptions']
            # we only want one instance per image for generation
            for d in descriptions[0:1]:
                arrays[0][d_idx, :, :] = self.format_sequence(d.split())
                if use_sourcelang:  # XXX ATTENTION this was originally self.hsn
                    #arrays[1][d_idx, 0, :] = self.get_hsn_features(split, data_key)
                    arrays[1][d_idx, 0, :] = self.get_source_features(split, data_key)
                if use_image:
                    # img_idx can be 1 or 2, depending on use_sourcelang
                    if self.args_dict.mrnn:
                        arrays[img_idx][d_idx, :, :] = self.get_image_features(
                            self.dataset, split, data_key)
                    else:
                        arrays[img_idx][d_idx, 0, :] = self.get_image_features(
                            self.dataset, split, data_key)
                d_idx += 1
            if d_idx >= intended_size:
                break

        # probably need to resize arrays for the generation data
        if self.args_dict.small_val or d_idx < split_size:
            # d_idx (number of descriptions before break) is new size.
            arrays = self.resize_arrays(d_idx, arrays)

        logger.debug("actual max_seq_len in split %s: %d",
                    split, self.actual_max_seq_len)
        logger.debug("dscrp_array size: %s", arrays[0].shape)
        # TODO: truncate dscrp_array, img_array, targets
        # to actual_max_seq_len (+ padding)

        return arrays

    def get_data_by_split(self, split, use_sourcelang=False, use_image=True):
        """ Gets all input data for model for a given split (ie. train, val,
        test).
        Returns tuple containing a list of training input arrays (depending on
        use_sourcelang and use_image) and a target array.
        Training arrays may contain (in this order)
            descriptions: input array for text LSTM [item, timestep,
                vocab_onehot] (necessary)
            source language:  input array of hidden state vectors for source
                language features (optional)
            image: input array of image features [item, timestep, img_feats at
                    timestep=0 else 0] (optional)
        Targets: target array for text LSTM (same format and data as
                descriptions, timeshifted)

        Changed: If small_val is set, the original arrays are now also a bit
        smaller than split_size (and then truncated to d_idx, as before). This
        is so I can run this on a lower-memory machine without thrashing.
        """

        if (self._cached_val_input is not None and split == "val"):
            logger.info("Retrieving cached val_input and val_targets")
            return (self._cached_val_input, self._cached_val_targets)

        logger.info("Making data for %s", split)
        if len(self.datasets) > 1 and split == "train":
            logger.warn("Called get_data_by_split on train while supertraining;\
                        this is probably NOT WHAT YOU INTENDED")

        split_size = self.split_sizes[split]
        intended_size = np.inf
        if self.args_dict.small_val:
            intended_size = SMALL_VAL
            # Make split_size comfortably bigger than intended_size
            split_size = 2 * SMALL_VAL * self.args_dict.num_sents

        arrays = self.get_new_training_arrays(split_size, use_sourcelang,
                                              use_image)
        if use_sourcelang and use_image:  # where is image array in arrays.
            img_idx = 2
        else:
            img_idx = 1

        d_idx = 0  # description index
        for data_key in self.dataset[split]:
            ds = self.dataset[split][data_key][self.ds_type]
            for description in ds:
                arrays[0][d_idx, :, :] = self.format_sequence(description.split())
                if use_sourcelang:  # XXX ATTENTION this was originally self.hsn
                    #arrays[1][d_idx, 0, :] = self.get_hsn_features(split, data_key)
                    arrays[1][d_idx, 0, :] = self.get_source_features(split, data_key)
                if use_image:
                    # img_idx can be 1 or 2, depending on use_sourcelang
                    if self.args_dict.mrnn:
                        arrays[img_idx][d_idx, :, :] = self.get_image_features(
                            self.dataset, split, data_key)
                    else:
                        arrays[img_idx][d_idx, 0, :] = self.get_image_features(
                            self.dataset, split, data_key)
                d_idx += 1
            if d_idx >= intended_size:
                break

        if self.args_dict.small_val:
            # d_idx (number of descriptions before break) is new size.
            arrays = self.resize_arrays(d_idx, arrays)

        targets = self.get_target_descriptions(arrays[0])

        logger.debug("actual max_seq_len in split %s: %d",
                    split, self.actual_max_seq_len)
        logger.debug("dscrp_array size: %s", arrays[0].shape)
        # TODO: truncate dscrp_array, img_array, targets
        # to actual_max_seq_len (+ padding)

        if split == "val":
            self._cached_val_input = arrays
            self._cached_val_targets = targets

        return (arrays, targets)

    def get_one_image_features_matrix(self, dataset, split, datakey):
        """ Creates the image features matrix/vector for a dataset split.
        Note that only the first (zero timestep) cell in the second dimension
        will be non-zero.
        """

        img_array = np.zeros((self.max_seq_len, IMG_FEATS))
        img_array[0, :] = self.get_image_features(dataset, split, datakey)
        return img_array

    def get_image_features_matrix(self, split):
        """ Creates the image features matrix/vector for a dataset split.
        Note that only the first (zero timestep) cell in the second dimension
        will be non-zero.
        """
        split_size = self.split_sizes[split]
        if self.small:
            split_size = 100

        img_array = np.zeros((split_size, self.max_seq_len, IMG_FEATS))
        for idx, data_key in enumerate(self.dataset[split]):
            if self.small and idx >= split_size:
                break
            img_array[idx, 0, :] = self.get_image_features(split, data_key)
        return img_array

    def get_hsn_features(self, split, data_key):
        """ Return image features vector for split[data_key]."""
        try:
            return self.source_dataset[split][data_key]['final_hidden_features']
        except KeyError:
            # this image -- description pair doesn't have a source-language
            # vector. Raise a KeyError so the requester can deal with the
            # missing data.
            logger.warning("Skipping '%s' because it doesn't have a source vector", data_key)
            raise KeyError

    def set_predicted_description(self, split, data_key, sentence):
        '''
        Set the predicted sentence tokens in the data_key group,
        creating the group if necessary, or erasing the current value if
        necessary.
        '''

        if self.openmode != "r+":
            # forcefully quit when trying to write to a read-only file
            raise RuntimeError("Dataset is read-only, try again with --h5_writable")

        dataset_key = 'predicted_description'

        try:
            predicted_text = self.dataset[split][data_key].create_dataset(dataset_key, (1,), dtype=h5py.special_dtype(vlen=unicode))
        except RuntimeError:
            # the dataset already exists, erase it and create an empty space
            del self.dataset[split][data_key][dataset_key]
            predicted_text = self.dataset[split][data_key].create_dataset(dataset_key, (1,), dtype=h5py.special_dtype(vlen=unicode))

        predicted_text[0] = " ".join([x for x in sentence])

    def set_source_features(self, split, data_key, dataset_key, feats, dims):
        '''
        Set the source feature vector stored in the dataset_key group,
        creating the group if necessary, or erasing the current value if
        necessary.
        '''

        if self.openmode != "r+":
            # forcefully quit when trying to write to a read-only file
            raise RuntimeError("Dataset is read-only, try again with --h5_writable")

        try:
            source_data = self.dataset[split][data_key].create_dataset(
                                  dataset_key, (dims,), dtype='float32')
        except RuntimeError:
            # the dataset already exists, erase it and create an empty space
            del self.dataset[split][data_key][dataset_key]
            source_data = self.dataset[split][data_key].create_dataset(
                                  dataset_key, (dims,), dtype='float32')

        source_data[:] = feats

    def get_source_features(self, split, data_key):
        '''
        Return the source feature vector from self.source_dataset.

        Relies on self.source_encoder,
                  self.source_dim,
                  self.source_type
        '''

        h5_dataset_str = "%s-hidden_feats-%s-%d" % (self.source_type,
                                                    self.source_encoder,
                                                    self.source_dim)
        try:
            return self.source_dataset[split][data_key][h5_dataset_str]
        except KeyError:
            # this image -- description pair doesn't have a source-language
            # vector. Raise a KeyError so the requester can deal with the
            # missing data.
            logger.warning("Skipping '%s' because it doesn't have a source vector", data_key)
            raise KeyError

    def get_image_features(self, dataset, split, data_key):
        """ Return image features vector for split[data_key]."""
        return dataset[split][data_key]['img_feats'][:]

    def set_vocabulary(self, path):
        '''
        Initialise the vocabulary from a checkpointed model.

        TODO: some duplication from extract_vocabulary
        '''
        self.extract_complete_vocab()
        logger.info("Initialising vocabulary from pre-defined model")
        try:
            v = cPickle.load(open("%s/../vocabulary.pk" % path, "rb"))
        except:
            v = cPickle.load(open("%s/vocabulary.pk" % path, "rb"))
        self.index2word = dict((v, k) for k, v in v.iteritems())
        self.word2index = dict((k, v) for k, v in v.iteritems())
        longest_sentence = 0
        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        self.longest_sentence = max(longest_sentence, train_longest, val_longest)
        self.corpus_statistics()
#        self.max_seq_len = longest_sentence + 2
#        logger.info("Max seq length %d, setting max_seq_len to %d",
#                    longest_sentence, self.max_seq_len)
#
#        logger.info("Split sizes %s", self.split_sizes)
#
#        logger.info("Number of words in vocabulary %d", len(self.word2index))
#        #logger.debug("word2index %s", self.word2index.items())
#        logger.debug("Number of indices %d", len(self.index2word))
#        #logger.debug("index2word: %s", self.index2word.items())

    def find_longest_sentence(self, split):
        '''
        Calculcates the length of the longest sentence in a given split of
        a dataset and updates the number of sentences in a split.
        TODO: can we get split_sizes from H5 dataset indices directly?
        '''
        local_ds_type = "descriptions" if split == 'train' else self.ds_type
        longest_sentence = 0
        for dataset in self.datasets:
            for data_key in dataset[split]:
                for description in dataset[split][data_key][local_ds_type][0:self.args_dict.num_sents]:
                    d = description.split()
                    if len(d) > longest_sentence:
                        longest_sentence = len(d)

        return longest_sentence

    def calculate_split_sizes(self):
        for split in ["train", "val", "test"]:
            for dataset in self.datasets:
                for data_key in dataset[split]:
                    for description in dataset[split][data_key]['descriptions'][0:self.args_dict.num_sents]:
                        self.split_sizes[split] += 1

    def extract_complete_vocab(self):
        self.unk_dict = defaultdict(int)
        for dataset in self.datasets:
            for data_key in dataset['train']:
                for description in dataset['train'][data_key]['descriptions'][0:self.args_dict.num_sents]:
                    for token in description.split():
                        self.unk_dict[token] += 1

    def extract_vocabulary(self):
        '''
        Collect word frequency counts over the train / val inputs and use
        these to create a model vocabulary. Words that appear fewer than
        self.unk times will be ignored.

        Also finds longest sentence, since it's already iterating over the
        whole dataset. HOWEVER this is the longest sentence *including* UNK
        words, which are removed from the data and shouldn't really be
        included in max_seq_len.
        But max_seq_len/longest_sentence is just supposed to be a safe
        upper bound, so we're good (except for some redundant cycles.)
        '''
        logger.info("Extracting vocabulary")
        self.extract_complete_vocab()

        longest_sentence = 0

        # set the length of the longest sentence
        train_longest = self.find_longest_sentence('train')
        val_longest = self.find_longest_sentence('val')
        self.longest_sentence = max(longest_sentence, train_longest, val_longest)

        # vocabulary is a word:id dict (superceded by/identical to word2index?)
        # <S>, <E> are special first indices
        vocabulary = {PAD: 0, BOS: 1, EOS: 2}
        for v in self.unk_dict:
            if self.unk_dict[v] > self.unk:
                vocabulary[v] = len(vocabulary)

        assert vocabulary[BOS] == 1
        assert vocabulary[EOS] == 2

        logger.info("Pickling dictionary to checkpoint/%s/vocabulary.pk",
                    self.run_string)
        try:
            os.mkdir("checkpoints/%s" % self.run_string)
        except OSError:
            pass
        cPickle.dump(vocabulary,
                     open("checkpoints/%s/vocabulary.pk"
                          % self.run_string, "wb"))

        self.index2word = dict((v, k) for k, v in vocabulary.iteritems())
        self.word2index = vocabulary
        self.corpus_statistics()

    def corpus_statistics(self):
        self.max_seq_len = self.longest_sentence + 2
        logger.info("Max seq length %d, setting max_seq_len to %d",
                    self.longest_sentence, self.max_seq_len)

        logger.info("Split sizes %s", self.split_sizes)

        logger.info("Number of words %d -> %d", len(self.unk_dict),
                    len(self.word2index))
        actual_len, true_len = self.discard_percentage()
        logger.info("Retained / Original Tokens: %d / %d (%.2f pc)",
                    actual_len, true_len, 100 * float(actual_len)/true_len)
        avg_len = self.avg_len()
        logger.info("Average train sentence length: %.2f tokens" % avg_len)
        #logger.debug("word2index %s", self.word2index.items())
        #logger.debug("Number of indices %d", len(self.index2word))
        #logger.debug("index2word: %s", self.index2word.items())

    def discard_percentage(self):
        '''
        One-off calculation of how many words are throw-out from the training
        sequences using the defined UNK threshold.
        '''
        true_len = 0
        actual_len = 0
        split = 'train'
        for data_key in self.dataset[split]:
            for description in self.dataset[split][data_key]['descriptions'][0:self.args_dict.num_sents]:
                d = description.split()
                true_len += len(d)
                unk_d = [self.word2index[w] for w in d if w in self.word2index]
                actual_len += len(unk_d)
        return (actual_len, true_len)

    def avg_len(self):
        '''
        One-off calculation of the average length of sentences in the training
        data before UNKing.
        '''
        true_len = 0
        num_sents = 0.0
        split = 'train'
        for data_key in self.dataset[split]:
            for description in self.dataset[split][data_key][self.ds_type][0:self.args_dict.num_sents]:
                d = description.split()
                true_len += len(d)
                num_sents += 1
        return (true_len/num_sents)

    def format_sequence(self, sequence):
        """ Transforms one sequence (description) into input matrix
        (timesteps, vocab-onehot)
        TODO: may need to add another (-1) timestep for image description
        TODO: add tokenization!
        """
        # zero default value equals padding
        seq_array = np.zeros((self.max_seq_len, len(self.word2index)))
        w_indices = [self.word2index[w] for w in sequence
                     if w in self.word2index]
        if len(w_indices) > self.actual_max_seq_len:
            self.actual_max_seq_len = len(w_indices)

        seq_array[0, self.word2index[BOS]] += 1  # BOS token at zero timestep
        time = 0
        for time, vocab in enumerate(w_indices):
            seq_array[time + 1, vocab] += 1
        # add EOS token at end of sentence
        try:
            assert time + 1 == len(w_indices),\
                "time %d sequence %s len w_indices %d seq_array %s" % (
                    time, " ".join([x for x in sequence]), len(w_indices),
                    seq_array)
        except AssertionError:
            if len(w_indices) == 0 and time == 0:
                # none of the words in this description appeared in the
                # vocabulary. this is most likely caused by the --unk
                # threshold.
                #
                # we don't encode this sentence because [BOS, EOS] doesn't
                # make sense
                logger.warning("Skipping '%s' because none of its words appear in the vocabulary" % ' '.join([x for x in sequence]))
                raise AssertionError
        seq_array[len(w_indices) + 1, self.word2index[EOS]] += 1
        return seq_array

    def get_target_descriptions(self, input_array):
        """ Target is always _next_ word, so we move input_array over by -1
        timestep (target at t=1 is input at t=2).
        """
        target_array = np.zeros(input_array.shape)
        target_array[:, :-1, :] = input_array[:, 1:, :]
        return target_array

    def get_target_array(self, input_array):
        """ Target is always _next_ word, so we move input_array over by -1
        timestep (target at t=1 is input at t=2).
        """
        target_array = np.zeros(input_array.shape)
        target_array[:-1, :] = input_array[1:, :]
        return target_array

    def random_generator(self, split):
        "Generator that produces input/output tuples for a given dataset and split."
        # For randomization, we use a independent Random instance.
        random_instance = random.Random()
        # Make sure that the desired split is actually in the dataset.
        assert split in self.dataset
        # Get a list of the keys. We will use this list to shuffle and iterate over.
        identifiers = self.dataset[split].keys()
        # Get the number of descriptions.
        first_id = identifiers[0]
        num_descriptions = len(self.dataset[split][first_id]['descriptions'])
        description_indices = list(range(num_descriptions))

        arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                              self.args_dict.source_vectors is not None,
                                              not self.args_dict.no_image)

        j = 0
        # Shuffle the description indices.
        random_instance.shuffle(description_indices)
        # And loop over them.
        while j <= len(identifiers):
            for desc_idx in description_indices:
                # For each iteration over the description indices, also shuffle the
                # identifiers.
                random_instance.shuffle(identifiers)
                # And loop over them.
                i = 0
                for ident in identifiers:
                    description = self.dataset[split][ident]['descriptions'][desc_idx]
                    img_feats = self.get_image_features(self.dataset, split, ident)
                    try:
                        description_array = self.format_sequence(description.split())
                        arrays[0][i] = description_array
                        if self.args_dict.mrnn:
                            arrays[1][i, :] = img_feats
                        else:
                            arrays[1][i][0] = img_feats
                        i += 1
                    except AssertionError:
                        # If the description doesn't share any words with the vocabulary.
                        pass
                    if i == self.args_dict.batch_size:
                        targets = self.get_target_descriptions(arrays[0])
                        yield {'text':arrays[0], 'img': arrays[1],
                               'output': targets}
                        i = 0
                        arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                              self.args_dict.source_vectors is not None,
                                              not self.args_dict.no_image)
            if i != 0:
                self.resize_arrays(i, arrays)
                targets = self.get_target_descriptions(arrays[0])
                #logger.info(' '.join([self.index2word[np.argmax(x)] for x in arrays[0][0,:,:]]))
                yield {'text':arrays[0], 'img': arrays[1],
                       'output': targets}
                i = 0
                j = 0
                arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                      self.args_dict.source_vectors is not None,
                                      not self.args_dict.no_image)

    def fixed_generator(self, split='val'):
        """Generator that returns the instances in a split in the fixed order
        defined in the underlying data. Useful for calculating perplexity, etc.
        No randomization."""
        arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                              self.args_dict.source_vectors is not None,
                                              not self.args_dict.no_image)
        i = 0
        j = 0
        # Get the number of descriptions.
        identifiers = self.dataset[split].keys()
        first_id = identifiers[0]
        num_descriptions = len(self.dataset[split][first_id]['descriptions'])
        description_indices = list(range(num_descriptions))

        while j <= len(identifiers):
            i = 0
            for ident in identifiers:
                for desc_idx in description_indices:
                    description = self.dataset[split][ident]['descriptions'][desc_idx]
                    img_feats = self.get_image_features(self.dataset, split, ident)
                    try:
                        description_array = self.format_sequence(description.split())
                        arrays[0][i] = description_array
                        if self.args_dict.mrnn:
                            arrays[1][i, :] = img_feats
                        else:
                            arrays[1][i][0] = img_feats
                        i += 1
                    except AssertionError:
                        # If the description doesn't share any words with the vocabulary.
                        logger.info('Could not encode %s', description)
                        pass
                if i == self.args_dict.batch_size:
                    targets = self.get_target_descriptions(arrays[0])
                    yield {'text':arrays[0], 'img': arrays[1],
                           'output': targets}
                    i = 0
                    arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                             self.args_dict.source_vectors is not None,
                                             not self.args_dict.no_image)
            if i != 0:
                logger.debug("Outside for loop")
                self.resize_arrays(i, arrays)
                targets = self.get_target_descriptions(arrays[0])
                logger.debug(' '.join([self.index2word[np.argmax(x)] for x in
                    arrays[0][0,:,:] if self.index2word[np.argmax(x)] != "<P>"]))
                yield {'text':arrays[0], 'img': arrays[1],
                       'output': targets}
                i = 0
                j = 0
                arrays = self.get_new_training_arrays(self.args_dict.batch_size,
                                      self.args_dict.source_vectors is not None,
                                      not self.args_dict.no_image)
#            for ident in identifiers:
#                description = self.dataset['val'][ident]['descriptions'][0]
#                img_feats = self.get_image_features(self.dataset, 'val', ident)
#                try:
#                    description_array = self.format_sequence(description.split())
#                    arrays[0][i] = description_array
#                    if self.args_dict.mrnn:
#                        arrays[1][i, :] = img_feats
#                    else:
#                        arrays[1][i][0] = img_feats
#                    i += 1
#                    j += 1
#                except AssertionError:
#                    # If the description doesn't share any words with the vocabulary.
#                    pass
#                if i == self.args_dict.batch_size:
#                    targets = self.get_target_descriptions(arrays[0])
#                    logger.debug(arrays[0].shape)
#                    logger.debug(' '.join([self.index2word[np.argmax(x)] for x
#                        in arrays[0][0,:,:] if self.index2word[np.argmax(x)]
#                        != "<P>"]))
#                    yield {'text':arrays[0], 'img': arrays[1],
#                           'output': targets}
#                    i = 0
#                    arrays = self.get_new_training_arrays(self.args_dict.batch_size,
#                                          self.args_dict.source_vectors is not None,
#                                                  not self.args_dict.no_image)

    def generation_generator(self, split='val', batch_size=-1):
        """Generator for generating descriptions.
        This will only return one array per instance in the data.
        No randomization.

        batch_size=1 will return minibatches of one instance. Use this for
        beam search decoding.
        """

        batch_size = self.args_dict.batch_size if batch_size == -1 else batch_size

        arrays = self.get_new_training_arrays(batch_size,
                                              self.args_dict.source_vectors is not None,
                                              not self.args_dict.no_image)
        i = 0
        j = 0
        identifiers = self.dataset[split].keys()

        for ident in identifiers:
           description = self.dataset[split][ident]['descriptions'][0]
           img_feats = self.get_image_features(self.dataset, split, ident)
           try:
               description_array = self.format_sequence(description.split())
               arrays[0][i] = description_array
               if self.args_dict.mrnn:
                   arrays[1][i, :] = img_feats
               else:
                   arrays[1][i][0] = img_feats
               i += 1
           except AssertionError:
               # If the description doesn't share any words with the vocabulary.
               pass
           if i == batch_size:
               targets = self.get_target_descriptions(arrays[0])
               logger.debug(arrays[0].shape)
               logger.debug(' '.join([self.index2word[np.argmax(x)] for x
                   in arrays[0][0,:,:] if self.index2word[np.argmax(x)]
                   != "<P>"]))
               yield {'text':arrays[0], 'img': arrays[1],
                      'output': targets, 'ident': ident}
               i = 0
               arrays = self.get_new_training_arrays(batch_size,
                                                     self.args_dict.source_vectors is not None,
                                                     not self.args_dict.no_image)
        if i != 0:
            logger.debug("Outside for loop")
            self.resize_arrays(i, arrays)
            targets = self.get_target_descriptions(arrays[0])
            logger.debug(' '.join([self.index2word[np.argmax(x)] for x in
                arrays[0][0,:,:] if self.index2word[np.argmax(x)] != "<P>"]))
            yield {'text':arrays[0], 'img': arrays[1],
                    'output': targets, 'ident': ident}
            i = 0
            arrays = self.get_new_training_arrays(batch_size,
                                                  self.args_dict.source_vectors is not None,
                                                  not self.args_dict.no_image)
