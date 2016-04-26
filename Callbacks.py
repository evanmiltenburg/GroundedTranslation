"""
Module to do callbacks for Keras models.
"""
from __future__ import division

from keras.callbacks import Callback  # ModelCheckpoint , EarlyStopping

import matplotlib.pyplot as plt
import h5py
import itertools
import logging
import numpy as np
import os
import subprocess
import shutil
import codecs
import sys
from time import gmtime, strftime
import math
import time
from copy import deepcopy

from memory_profiler import profile

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096
HSN_SIZE = 409


class CompilationOfCallbacks(Callback):
    """ Collection of compiled callbacks."""

    def __init__(self, word2index, index2word, argsDict, dataset,
                 data_generator, use_sourcelang=False, use_image=True):
        super(Callback, self).__init__()

        self.verbose = True
        self.filename = "weights.hdf5"
        self.save_best_only = True

        self.val_loss = []
        self.best_val_loss = np.inf

        self.val_pplx = []
        self.best_val_pplx = np.inf

        self.val_bleu = []
        self.best_val_bleu = np.NINF

        self.word2index = word2index
        self.index2word = index2word
        self.args = argsDict

        # used to control early stopping on the validation data
        self.wait = 0
        self.patience = self.args.patience

        # needed by model.predict in generate_sentences
        self.use_sourcelang = use_sourcelang
        self.use_image = use_image

        # controversial assignment but it makes it much easier to
        # perform pplx calculations
        self.data_generator = data_generator

        # this results in two file handlers for dataset (here and
        # data_generator)
        if not dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % dataset, "r")
        if self.args.source_vectors is not None:
            self.source_dataset = h5py.File("%s/dataset.h5" % self.args.source_vectors, "r")

    def on_epoch_end(self, epoch, logs={}):
        '''
        At the end of each epoch we
          1. create a directory to checkpoint data
          2. save the arguments used to initialise the run
          3. generate N sentences in the val data by sampling from the model
          4. calculate BLEU score of the generated sentences
          5. determine whether to stop training and sys.exit(0)
          6. save the model parameters using BLEU
        '''
        savetime = strftime("%d%m%Y-%H%M%S", gmtime())
        path = self.create_checkpoint_directory(savetime)
        self.save_run_arguments(path)

        # Generate training and val sentences to check for overfitting
        if self.args.enable_val_pplx:
            val_pplx = self.calculate_pplx(path)
        self.generate_sentences(path)
        val_bleu = self.__bleu_score__(path)

        self.early_stop_decision(len(self.val_bleu)+1, val_bleu, val_pplx)
        self.checkpoint_parameters(epoch, logs, path, val_bleu, val_pplx)
        self.log_performance()

    def early_stop_decision(self, epoch, val_bleu, val_pplx):
        '''
        Stop training if validation BLEU score has not increased for
        --patience epochs.

        WARNING: quits with sys.exit(0).
        '''

        if val_bleu > self.best_val_bleu or self.args.no_early_stopping:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                # we have exceeded patience
                if val_pplx > self.best_val_pplx:
                    # and pplx is no longer decreasing
                    logger.info("Epoch %d: early stopping", epoch)
                    handle = open("checkpoints/%s/summary"
                                  % self.args.run_string, "a")
                    handle.write("Early stopping because patience exceeded\n")
                    handle.close()
                    sys.exit(0)

    def log_performance(self):
        '''
        Record model performance so far, based on whether we are tracking
        validation loss or validation pplx.
        '''
        handle = open("checkpoints/%s/summary" % self.args.run_string, "w")

        for epoch in range(len(self.val_pplx)):
            if self.args.enable_val_pplx:
                logger.info("Checkpoint %d | val pplx: %.5f bleu %.2f",
                            epoch+1, self.val_pplx[epoch],
                            self.val_bleu[epoch])
                handle.write("Checkpoint %d | val pplx: %.5f bleu %.2f\n"
                             % (epoch+1, self.val_pplx[epoch],
                                self.val_bleu[epoch]))
            else:
                logger.info("Checkpoint %d | val loss: %.5f bleu %.2f",
                            epoch+1, self.val_loss[epoch],
                            self.val_bleu[epoch])
                handle.write("Checkpoint %d | val loss: %.5f bleu %.2f\n"
                             % (epoch+1, self.val_loss[epoch],
                                self.val_bleu[epoch]))

        logger.info("---")  # break up the presentation for clarity

        # BLEU is the quickest indicator of performance for our task
        # but PPLX (2^loss) is our objective function
        best_bleu = np.nanargmax(self.val_bleu)
        if self.args.enable_val_pplx:
            best_pplx = np.nanargmin(self.val_pplx)
            logger.info("Best BLEU: %d | val pplx %.5f bleu %.2f",
                        best_bleu+1, self.val_pplx[best_bleu],
                        self.val_bleu[best_bleu])
            handle.write("Best BLEU: %d | val pplx %.5f bleu %.2f\n"
                         % (best_bleu+1, self.val_pplx[best_bleu],
                            self.val_bleu[best_bleu]))
            logger.info("Best PPLX: %d | val pplx %.5f bleu %.2f",
                        best_pplx+1, self.val_pplx[best_pplx],
                        self.val_bleu[best_pplx])
            handle.write("Best PPLX: %d | val pplx %.5f bleu %.2f\n"
                         % (best_pplx+1, self.val_pplx[best_pplx],
                            self.val_bleu[best_pplx]))
        else:
            logger.info("Best checkpoint: %d | val loss %.5f bleu %.2f",
                        best_bleu+1, self.val_loss[best_bleu],
                        self.val_bleu[best_bleu])
            handle.write("Best checkpoint: %d | val loss %.5f bleu %.2f"
                         % (best_bleu+1, self.val_loss[best_bleu],
                            self.val_bleu[best_bleu]))
        logger.info("Early stopping marker: wait/patience: %d/%d",
                    self.wait, self.patience)
        handle.write("Early stopping marker: wait/patience: %d/%d\n" %
                     (self.wait, self.patience))
        handle.close()

    def extract_references(self, directory, split):
        """
        Get reference descriptions for val or test data.
        """
        references = self.data_generator.get_refs_by_split_as_list(split)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d' % (directory, split, refid),
                        'w', 'utf-8').write('\n'.join([x[refid] for x in references]))
                        #'w', 'utf-8').write('\n'.join(['\n'.join(x) for x in references]))
        return references

    def __bleu_score__(self, directory, val=True):
        '''
        PPLX is only weakly correlated with improvements in BLEU,
        and thus improvements in human judgements. Let's also track
        BLEU score of a subset of generated sentences in the val split
        to decide on early stopping, etc.
        '''

        prefix = "val" if val else "test"

        self.extract_references(directory, split=prefix)

        subprocess.check_call(
            ['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated > %s/%sBLEU'
             % (directory, prefix, directory, prefix, directory, prefix)],
            shell=True)
        bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
        data = bleudata.split(",")[0]
        bleuscore = data.split("=")[1]
        bleu = float(bleuscore.lstrip())
        return bleu

    def create_checkpoint_directory(self, savetime):
        '''
        We will create one directory to store all of the epochs data inside.
        The name is based on the run_string (if provided) or the current time.
        '''

        prefix = self.args.run_string if self.args.run_string != "" else ""
        number = "%03d" % (len(self.val_bleu) + 1)
        filepath = "checkpoints/%s/%s-%s" % ((prefix, number, savetime))
        try:
            os.mkdir("checkpoints/%s/" % (prefix))
            shutil.copyfile("train.py", "checkpoints/%s/train.py" % prefix)
            shutil.copyfile("models.py", "checkpoints/%s/models.py" % prefix)
        except OSError:
            pass  # directory already exists
        try:
            os.mkdir(filepath)
        except OSError:
            pass  # directory already exists
        logger.info("In %s ...",filepath)
        return filepath

    def save_run_arguments(self, filepath):
        '''
        Save the command-line arguments, along with the method defaults,
        used to parameterise this run.
        '''
        handle = open("%s/argparse.args" % filepath, "w")
        for arg, value in self.args.__dict__.iteritems():
            handle.write("%s: %s\n" % (arg, str(value)))
        handle.close()

    def checkpoint_parameters(self, epoch, logs, filepath, cur_val_bleu,
                              cur_val_pplx=0.):
        '''
        We checkpoint the model parameters based on either PPLX reduction or
        BLEU score increase in the validation data. This is driven by the
        user-specified argument self.args.stopping_loss.
        '''

        weights_path = "%s/weights.hdf5" % filepath

        if self.save_best_only and self.params['do_validation']:
            cur_val_loss = logs.get('val_loss')
            self.val_loss.append(cur_val_loss)
            if cur_val_loss < self.best_val_loss:
                self.best_val_loss = cur_val_loss

            #logger.info("Checkpoint %d: | val loss %0.5f (best: %0.5f) bleu %0.2f (best %0.2f)", 
            #            (len(self.val_loss) + 1),
            #            cur_val_loss, self.best_val_loss, cur_val_bleu,
            #            self.best_val_bleu)

        if self.args.enable_val_pplx:
            #logger.info("Checkpoint %d: | val pplx %0.5f (best: %0.5f) bleu %0.2f (best %0.2f)", 
            #            (len(self.val_pplx) + 1),
            #            cur_val_pplx, self.best_val_pplx, cur_val_bleu,
            #            self.best_val_bleu)
            self.val_pplx.append(cur_val_pplx)
            if cur_val_pplx < self.best_val_pplx:
                self.best_val_pplx = cur_val_pplx

        # save the weights anyway for debug purposes
        self.model.save_weights(weights_path, overwrite=True)

        # update the best values, if applicable
        self.val_bleu.append(cur_val_bleu)
        if cur_val_bleu > self.best_val_bleu:
            self.best_val_bleu = cur_val_bleu

        optimiser_params = open("%s/optimiser_params" % filepath, "w")
        for key, value in self.model.optimizer.get_config().items():
            optimiser_params.write("%s: %s\n" % (key, value))
        optimiser_params.close()

    def yield_chunks(self, len_split_indices, batch_size):
        '''
        self.args.batch_size is not always cleanly divisible by the number of
        items in the split, so we need to always yield the correct number of
        items.
        '''
        for i in xrange(0, len_split_indices, batch_size):
            # yield split_indices[i:i+batch_size]
            yield (i, i+batch_size-1)

    def make_generation_arrays(self, prefix, fixed_words, generation=False):
        """Create arrays that are used as input for generation. """

        input_data =\
            self.data_generator.get_generation_data_by_split(prefix,
                                self.use_sourcelang, self.use_image)

        # Replace input words (input_data[0]) with zeros for generation,
        # except for the first args.generate_from_N_words
        # NOTE: this will include padding and BOS steps (fixed_words has been
        # incremented accordingly already in generate_sentences().)
        logger.info("Initialising with the first %d gold words (incl BOS)",
                    fixed_words)
        gen_input_data = deepcopy(input_data)
        gen_input_data[0][:, fixed_words:, :] = 0

        return gen_input_data

    def reset_text_arrays(self, text_arrays, fixed_words=1):
        """ Reset the values in the text data structure to zero so we cannot
        accidentally pass them into the model """
        reset_arrays = deepcopy(text_arrays)
        reset_arrays[:,fixed_words:, :] = 0
        return reset_arrays

    def generate_sentences(self, filepath, val=True):
        """
        Generates descriptions of images for --generation_timesteps
        iterations through the LSTM. Each input description is clipped to
        the first <BOS> token, or, if --generate_from_N_words is set, to the
        first N following words (N + 1 BOS token).
        This process can be additionally conditioned
        on source language hidden representations, if provided by the
        --source_vectors parameter.
        The output is clipped to the first EOS generated, if it exists.

        TODO: duplicated method with generate.py
        """
        prefix = "val" if val else "test"
        logger.info("Generating %s descriptions", prefix)
        start_gen = self.args.generate_from_N_words + 1  # include BOS
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), 
                             "w", 'utf-8')

        val_generator = self.data_generator.generation_generator(prefix)
        seen = 0
        for data in val_generator:
            text = data['text']
            # Append the first start_gen words to the complete_sentences list
            # for each instance in the batch.
            complete_sentences = [[] for _ in range(text.shape[0])]
            for t in range(start_gen):  # minimum 1
                for i in range(text.shape[0]):
                    w = np.argmax(text[i, t])
                    complete_sentences[i].append(self.index2word[w])
            text = self.reset_text_arrays(text, start_gen)
            img = data['img']
            Y_target = data['output']

            for t in range(start_gen, self.args.generation_timesteps):
                logger.debug("Input token: %s" % self.index2word[np.argmax(text[0,t-1])])
                preds = self.model.predict({'text': text,
                                            'img':  img},
                                            verbose=0)

                # Look at the last indices for the words.
                next_word_indices = np.argmax(preds['output'][:, t-1], axis=1)
                logger.debug("Predicted token: %s" % self.index2word[next_word_indices[0]])
                # update array[0]/sentence-so-far with generated words.
                for i in range(len(next_word_indices)):
                    text[i, t, next_word_indices[i]] = 1.
                next_words = [self.index2word[x] for x in next_word_indices]
                for i in range(len(next_words)):
                    complete_sentences[i].append(next_words[i])

            sys.stdout.flush()
            # print/extract each sentence until it hits the first end-of-string token
            for s in complete_sentences:
                decoded_str = ' '.join([x for x
                                        in itertools.takewhile(
                                            lambda n: n != "<E>", s[1:])])
                handle.write(decoded_str + "\n")

            seen += text.shape[0]
            if seen == self.data_generator.split_sizes['val']:
                # Hacky way to break out of the generator
                break
        handle.close()

    def calculate_pplx_old(self, path, val=True):
        """ Without batching. Robust against multiple descriptions/image,
        since it uses data_generator.get_data_by_split input. 
        We ignore OOV tokens, in line with other toolkits. See (D1) e.g.
        http://www.speech.sri.com/projects/srilm/manpages/srilm-faq.7.html

        Updated to prepare the Y_target data from the raw text instead of
        bundling it up as a 1-hot vector.
        """
        prefix = "val" if val else "test"
        logger.info("Calculating pplx over %s data", prefix)
        sum_logprobs = 0.0
        y_len = 0

        input_data, Y_target = self.data_generator.get_data_by_split(prefix,
                                       self.use_sourcelang, self.use_image)

        if self.args.debug:
            tic = time.time()

        preds = self.model.predict({'text': input_data[0],
                                    'img': input_data[1]}, verbose=0)

        if self.args.debug:
            logger.info("Forward pass took %f", time.time()-tic)

        for t in range(Y_target.shape[1]):
            for i in range(Y_target.shape[0]):
                target_idx = np.argmax(Y_target[i, t])
                if self.index2word[target_idx] != "<P>":
                    log_p = math.log(preds['output'][i, t, target_idx],2)
                    sum_logprobs += -log_p
                    y_len += 1

        norm_logprob = sum_logprobs / y_len
        pplx = math.pow(2, norm_logprob)
        logger.info("PPLX: %.4f", pplx)
        return pplx

    def calculate_pplx(self, val=True):
        """ Splits the input data into batches of self.args.batch_size to
        reduce the memory footprint of holding all of the data in RAM. """

        prefix = "val" if val else "test"
        logger.info("Calculating pplx over %s data", prefix)
        sum_logprobs = 0
        y_len = 0

        val_generator = self.data_generator.fixed_generator(prefix)
        seen = 0
        for data in val_generator:
            text = data['text']
            img = data['img']
            Y_target = data['output']

            preds = self.model.predict({'text': text, 'img': img},
                                       verbose=0,
                                       batch_size=self.args.batch_size)

            for i in range(Y_target.shape[0]):
                for t in range(Y_target.shape[1]):
                    target_idx = np.argmax(Y_target[i, t])
                    target_tok = self.index2word[target_idx]
                    if target_tok != "<P>":
                        log_p = math.log(preds['output'][i, t, target_idx],2)
                        sum_logprobs += -log_p
                        y_len += 1

            seen += text.shape[0]
            if seen == self.data_generator.split_sizes['val']:
                # Hacky way to break out of the generator
                break

        norm_logprob = sum_logprobs / y_len
        pplx = math.pow(2, norm_logprob)
        logger.info("PPLX: %.4f", pplx)
        return pplx
