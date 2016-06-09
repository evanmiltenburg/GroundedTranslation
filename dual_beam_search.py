from __future__ import print_function
import json
import numpy as np
np.set_printoptions(threshold='nan')
import h5py
import theano

import argparse
import itertools
import subprocess
import logging
import time
import codecs
import os
from copy import deepcopy
import math
import sys

from data_generator import VisualWordDataGenerator
import models

# Set up logger
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dimensionality of image feature vector
IMG_FEATS = 4096
MULTEVAL_DIR = '../multeval-0.5.1' if "util" in os.getcwd() else "multeval-0.5.1"

class cd:
    """Context manager for changing the current working directory"""
    """http://stackoverflow.com/questions/431684/how-do-i-cd-in-python"""
    def __init__(self, newPath):
        self.newPath = newPath

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class GroundedTranslationGenerator:

    def __init__(self, args):
        self.args = args
        self.vocab = dict()
        self.unkdict = dict()
        self.counter = 0
        self.maxSeqLen = 0

        # consistent with models.py
        self.use_sourcelang = args.source_vectors is not None
        self.use_image = not args.no_image
        self.model = None
        self.prepare_datagenerator()

        # this results in two file handlers for dataset (here and
        # data_generator)
        if not self.args.dataset:
            logger.warn("No dataset given, using flickr8k")
            self.dataset = h5py.File("flickr8k/dataset.h5", "r")
        else:
            self.dataset = h5py.File("%s/dataset.h5" % self.args.dataset, "r")

        if self.args.debug:
            theano.config.optimizer = 'None'
            theano.config.exception_verbosity = 'high'

    def prepare_datagenerator(self):
        self.data_gen = VisualWordDataGenerator(self.args,
                                                self.args.dataset)
        self.args.checkpoint = self.find_best_checkpoint()
        self.data_gen.set_vocabulary(self.args.checkpoint)
        self.vocab_len = len(self.data_gen.index2word)
        self.index2word = self.data_gen.index2word
        self.word2index = self.data_gen.word2index

    def generate(self):
        '''
        Entry point for this module.
        Loads up a data generator to get the relevant image / source features.
        Builds the relevant model, given the command-line arguments.
        Generates sentences for the images in the val / test data.
        Calculates BLEU and PPLX, unless requested.
        '''

        if self.use_sourcelang:
            # HACK FIXME unexpected problem with input_data
            self.hsn_size = self.data_gen.hsn_size
        else:
            self.hsn_size = 0

        if self.model == None:
            self.build_model(generate=True)

        self.generate_sentences(self.args.checkpoint, val=not self.args.test)
        if not self.args.without_scores:
            score = self.bleu_score(self.args.checkpoint, val=not self.args.test)
            if self.args.multeval:
                score, _, _ = self.multeval_scores(self.args.checkpoint,
                                                    val=not self.args.test)
            if not self.args.no_pplx:
                self.build_model(generate=False)
                self.calculate_pplx(self.args.checkpoint, val=not self.args.test)
            return score

################################################################################
# Helper functions for generate_sentences()

    def get_keep_func(self):
        "Builds a keep function, given a JSON file with info on what to keep."
        with open(self.args.keep_file) as f:
            d = json.load(f)
        
        whole_word = set(d['WHOLEWORD'])
        prefixes = d['STARTSWITH']
        suffixes = d['ENDSWITH']
        
        def keep_func(word):
            "Function to determine which words to keep in the beam."
            if word in whole_word:
                return True
            for pref in prefixes:
                if word.startswith(pref):
                    return True
            for suf in suffixes:
                if word.endswith(suf):
                    return True
            return False
        # Return the function:
        return keep_func

    def get_candidates(self, t, beams, structs, keep_func):
        """
        Get candidate beams containing the next word. If the next word is one that
        should be kept according to keep_func, the beams will be added to kept_candidates.
        """
        # Store the candidates produced at timestep t, will be
        # pruned at the end of the timestep
        candidates = []
        kept_candidates = []

        # we take a view of the datastructures, which means we're only
        # ever generating a prediction for the next word. This saves a
        # lot of cycles.
        preds = self.model.predict(structs, verbose=0)

        # The last indices in preds are the predicted words
        next_word_indices = preds['output'][:, t-1]
        sorted_indices = np.argsort(-next_word_indices, axis=1)

        # Each instance in structs is holding the history of a
        # beam, and so there is a direct connection between the
        # index of a beam in beams and the index of an instance in
        # structs.
        for beam_idx, b in enumerate(beams):
            # get the sorted predictions for the beam_idx'th beam
            beam_predictions = sorted_indices[beam_idx]
            for top_idx in range(self.args.beam_width):
                wordIndex = beam_predictions[top_idx]
                wordProb = next_word_indices[beam_idx][beam_predictions[top_idx]]
                # For the beam_idxth beam, add the log probability
                # of the top_idxth predicted word to the previous
                # log probability of the sequence, and  append the
                # top_idxth predicted word to the sequence of words
                current_word = self.index2word[wordIndex]
                updated_beam = [b[0] + math.log(wordProb), b[1] + [wordIndex]]
                candidates.append(updated_beam)
                if keep_func(current_word):
                    logger.info("WORD KEPT: " + current_word)
                    self.found_negation = True
                    kept_candidates.append(updated_beam)
        return candidates, kept_candidates

    def prune(self, candidates, max_beam_width, category='regular beams'):
        """
        Prune the candidates, so that we are left with max_beam_width
        beams. Also return beams that are finished as a separate list.
        """
        beams = candidates[:max_beam_width] # prune the beams
        finished = []
        pruned = []
        for b in beams:
            # If a top candidate emitted an EOS token then
            # a) add it to the list of finished sequences
            # b) remove it from the beams and decrease the
            # maximum size of the beams.
            if b[1][-1] == self.word2index["<E>"]:
                finished.append(b)
                if max_beam_width >= 1:
                    max_beam_width -= 1
            else:
                pruned.append(b)
        
        beams = pruned[:max_beam_width]

        if self.args.verbose:
            logger.info("Pruned beams " + ''.join(['(', category, ')']))
            logger.info("---")
            for b in beams:
                logger.info(" ".join([self.index2word[x] for x in b[1]]) + "(%f)" % b[0])
        return beams, finished

    def get_structs(self, beams, data, max_beam_width):
        "Get structs for the next round."
        structs = self.make_duplicate_matrices(data, max_beam_width)
        # Rewrite the 1-hot word features with the
        # so-far-predcicted tokens in a beam.
        for bidx, b in enumerate(beams):
            for idx, w in enumerate(b[1]):
                # This variable doesn't do anything.
                # next_word_index = w
                structs['text'][bidx, idx+1, w] = 1.
        return structs

    def log_finished(self, finished, category='regular beams'):
        "Log the Length-normalised samples."
        logger.info("Length-normalised samples" + ''.join(['(', category, ')']))
        logger.info("---")
        for f in finished:
            logger.info(" ".join([self.index2word[x] for x in f[1]]) + "(%f)" % f[0])

################################################################################


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
        try:
            assert self.args.beam_width > 1
        except AssertionError:
            raise AssertionError('Beam size too small. Cannot use dual beam search.')
        
        neg_counter = 0
        ident_desc_dict = dict()
        keep_func = self.get_keep_func()
        
        prefix = "val" if val else "test"
        handle = codecs.open("%s/%sGenerated" % (filepath, prefix), "w",
                             'utf-8')
        logger.info("Generating %s descriptions", prefix)

        start_gen = self.args.generate_from_N_words  # Default 0
        start_gen = start_gen + 1  # include BOS

        generator = self.data_gen.generation_generator(prefix, batch_size=1)

        # we are going to beam search for the most probably sentence.
        # let's do this one sentence at a time to make the logging output
        # easier to understand
        for seen, data in enumerate(generator, start=1):
            text = data['text']
            # Append the first start_gen words to the complete_sentences list
            # for each instance in the batch.
            complete_sentences = [[] for _ in range(text.shape[0])]
            for t in range(start_gen):  # minimum 1
                for i in range(text.shape[0]):
                    w = np.argmax(text[i, t])
                    complete_sentences[i].append(self.index2word[w])
            del data['text']
            text = self.reset_text_arrays(text, start_gen)
            Y_target = data['output']
            data['text'] = text

            max_beam_width = self.args.beam_width
            neg_max_beam_width = self.args.beam_width
            structs = self.make_duplicate_matrices(data, max_beam_width)

            # A beam is a 2-tuple with the probability of the sequence and
            # the words in that sequence. Start with empty beams
            beams = [(0.0, [])]
            neg_beams = []
            # collects beams that are in the top candidates and
            # emitted a <E> token.
            finished = []
            neg_finished = []
            # Flag variable. Is set to True once the first negation is found.
            self.found_negation = False
            
            for t in range(start_gen, self.args.generation_timesteps):
                # Ensure that kept_candidates is there. (And that previous results are removed.)
                kept_candidates = []
                
                ################################################################
                # GET CANDIDATES
                
                if max_beam_width > 0:
                    candidates, kept_candidates = self.get_candidates(t, beams, structs, keep_func)
                    candidates.sort(reverse = True)
                
                if self.found_negation:
                    neg_c, neg_kc = self.get_candidates(t, neg_beams, neg_structs, keep_func)
                    # don't add neg_kc: don't add examples twice.
                    neg_candidates = kept_candidates + neg_c
                    neg_candidates.sort(reverse = True)
                
                ################################################################
                # LOG NEW CANDIDATES
                
                if self.args.verbose:
                    logger.info("Candidates in the beam")
                    logger.info("---")
                    if max_beam_width > 0:
                        logger.info("REGULAR BEAM:")
                        for c in candidates:
                            logger.info(" ".join([self.index2word[x] for x in c[1]]) + " (%f)" % c[0])
                    if self.found_negation:
                        logger.info("SEPARATE BEAM:")
                        for c in neg_candidates:
                            logger.info(" ".join([self.index2word[x] for x in c[1]]) + " (%f)" % c[0])

                ################################################################
                # PRUNE
                
                beams, finished_this_round = self.prune(candidates,
                                                        max_beam_width,
                                                        category='regular beams')
                finished.extend(finished_this_round)
                if self.found_negation:
                    neg_beams, finished_this_round = self.prune(neg_candidates,
                                                                neg_max_beam_width,
                                                                category='selected beams')
                    neg_finished.extend(finished_this_round)
                

                ################################################################
                # STOP DECISION

                if self.found_negation:
                    if neg_max_beam_width == 0:
                        # We have sampled neg_max_beam_width sequences with an <E>
                        # token so stop the beam search.
                        break
                elif max_beam_width == 0:
                    # We have sampled max_beam_width sequences with an <E>
                    # token so stop the beam search.
                    break

                ################################################################
                # UPDATE STRUCTS

                # Reproduce the structs for the beam search so we can keep
                # track of the state of each beam
                if max_beam_width > 0:
                    structs = self.get_structs(beams=beams,
                                               data=data,
                                               max_beam_width=max_beam_width)
                
                neg_structs = self.get_structs(beams=neg_beams,
                                               data=data,
                                               max_beam_width=neg_max_beam_width)

            ####################################################################
            # WRAPPING UP

            # If none of the sentences emitted an <E> token while
            # decoding, add the final beams into the final candidates
            if len(finished) == 0:
                for leftover in beams:
                    finished.append(leftover)
            
            # Do the same for the neg beams.
            if self.found_negation and len(neg_finished) == 0:
                for leftover in neg_beams:
                    neg_finished.append(leftover)

            # Normalise the probabilities by the length of the sequences
            # as suggested by Graves (2012) http://arxiv.org/abs/1211.3711
            for f in finished:
                f[0] = f[0] / len(f[1])
            finished.sort(reverse=True)

            for f in neg_finished:
                f[0] = f[0] / len(f[1])
            neg_finished.sort(reverse=True)

            ####################################################################
            # LOG FINISHED
            
            if self.args.verbose:
                self.log_finished(finished, category='regular beams')
                if self.found_negation:
                    self.log_finished(finished, category='selected beams')

            # Emit the lowest (log) probability sequence
            best_beam = finished[0] if not self.found_negation else neg_finished[0]
            complete_sentences[i] = [self.index2word[x] for x in best_beam[1]]
            generated_sentence = ' '.join([x for x
                                   in itertools.takewhile(
                                       lambda n: n != "<E>", complete_sentences[i])])
            
            handle.write(generated_sentence + "\n")
            if self.args.verbose:
                logger.info("%s (%f)", generated_sentence, best_beam[0])
            
            if self.found_negation:
                neg_counter += 1
            
            if seen == self.data_gen.split_sizes[prefix]:
                # Hacky way to break out of the generator
                break
        
        logger.info("Total number of kept sentences: " + str(neg_counter))
        handle.close()

    def calculate_pplx(self, path, val=True):
        """ Splits the input data into batches of self.args.batch_size to
        reduce the memory footprint of holding all of the data in RAM. """

        prefix = "val" if val else "test"
        logger.info("Calculating pplx over %s data", prefix)
        sum_logprobs = 0
        y_len = 0

        generator = self.data_gen.fixed_generator(prefix)
        seen = 0
        for data in generator:
            Y_target = deepcopy(data['output'])
            del data['output']

            preds = self.model.predict(data,
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

            seen += data['text'].shape[0]
            if seen == self.data_gen.split_sizes[prefix]:
                # Hacky way to break out of the generator
                break

        norm_logprob = sum_logprobs / y_len
        pplx = math.pow(2, norm_logprob)
        logger.info("PPLX: %.4f", pplx)
        handle = open("%s/%sPPLX" % (path, prefix), "w")
        handle.write("%f\n" % pplx)
        handle.close()
        return pplx


    def reset_text_arrays(self, text_arrays, fixed_words=1):
        """ Reset the values in the text data structure to zero so we cannot
        accidentally pass them into the model.

        Helper function for generate_sentences().
         """
        reset_arrays = deepcopy(text_arrays)
        reset_arrays[:,fixed_words:, :] = 0
        return reset_arrays

    def make_duplicate_matrices(self, generator_data, k):
        '''
        Prepare K duplicates of the input data for a given instance yielded by
        the data generator.

        Helper function for the beam search decoder in generation_sentences().
        '''

        if self.use_sourcelang and self.use_image:
            # the data generator yielded a dictionary with the words, the
            # image features, and the source features
            dupes = [[],[],[]]
            words = generator_data['text']
            img = generator_data['img']
            source = generator_data['source']
            for x in range(k):
                # Make a deep copy of the word_feats structures
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(source[0,:,:])
                dupes[2].append(img[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])
            dupes[2] = np.array(dupes[2])

            return {'text': dupes[0], 'img': dupes[2], 'source': dupes[1]}

        elif self.use_image:
            # the data generator yielded a dictionary with the words and the
            # image features
            dupes = [[],[]]
            words = generator_data['text']
            img = generator_data['img']
            for x in range(k):
                # Make a deep copy of the word_feats structures
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(img[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])

            return {'text': dupes[0], 'img': dupes[1]}

        elif self.use_sourcelang:
            # the data generator yielded a dictionary with the words and the
            # source features
            dupes = [[],[]]
            words = generator_data['text']
            source= generator_data['source']
            for x in range(k):
                # Make a deep copy of the word_feats structures
                # so the arrays will never be shared
                dupes[0].append(deepcopy(words[0,:,:]))
                dupes[1].append(source[0,:,:])

            # Turn the list of arrays into a numpy array
            dupes[0] = np.array(dupes[0])
            dupes[1] = np.array(dupes[1])

            return {'text': dupes[0], 'source': dupes[1]}

    def find_best_checkpoint(self):
        '''
        Read the summary file from the directory and scrape out the run ID of
        the highest BLEU scoring checkpoint. Then do an ls-stlye function in
        the directory and return the exact path to the best model.

        Assumes only one matching prefix in the model checkpoints directory.
        '''

        summary_data = open("%s/summary" % self.args.model_checkpoints).readlines()
        summary_data = [x.replace("\n", "") for x in summary_data]
        best_id = None
        target = "Best Metric" if self.args.best_pplx else "Best loss"
        for line in summary_data:
            if line.startswith(target):
                best_id = "%03d" % (int(line.split(":")[1].split("|")[0]))

        checkpoint = None
        if best_id is not None:
            checkpoints = os.listdir(self.args.model_checkpoints)
            for c in checkpoints:
                if c.startswith(best_id):
                    checkpoint = c
                    break
        logger.info("Best checkpoint: %s/%s" % (self.args.model_checkpoints, checkpoint))
        return "%s/%s" % (self.args.model_checkpoints, checkpoint)

    def bleu_score(self, directory, val=True):
        '''
        PPLX is only weakly correlated with improvements in BLEU,
        and thus improvements in human judgements. Let's also track
        BLEU score of a subset of generated sentences in the val split
        to decide on early stopping, etc.
        '''

        prefix = "val" if val else "test"
        self.extract_references(directory, val)

        subprocess.check_call(
            ['perl multi-bleu.perl %s/%s_reference.ref < %s/%sGenerated | tee %s/%sBLEU'
             % (directory, prefix, directory, prefix, directory, prefix)], shell=True)
        bleudata = open("%s/%sBLEU" % (directory, prefix)).readline()
        data = bleudata.split(",")[0]
        bleuscore = data.split("=")[1]
        bleu = float(bleuscore.lstrip())
        return bleu

    def multeval_scores(self, directory, val=True):
        '''
        Maybe you want to evaluate with Meteor, TER, and BLEU?
        '''
        prefix = "val" if val else "test"
        self.extract_references(directory, val)

        with cd(MULTEVAL_DIR):
            subprocess.check_call(
                ['./multeval.sh eval --refs ../%s/%s_reference.* \
                 --hyps-baseline ../%s/%sGenerated \
                 --meteor.language de \
		 --threads 4 \
		2> multevaloutput 1> multevaloutput'
                % (directory, prefix, directory, prefix)], shell=True)
            handle = open("multevaloutput")
            multdata = handle.readlines()
            handle.close()
            for line in multdata:
              if line.startswith("RESULT: baseline: BLEU: AVG:"):
                mbleu = line.split(":")[4]
                mbleu = mbleu.replace("\n","")
                mbleu = mbleu.strip()
                lr = mbleu.split(".")
                mbleu = float(lr[0]+"."+lr[1][0:2])
              if line.startswith("RESULT: baseline: METEOR: AVG:"):
                mmeteor = line.split(":")[4]
                mmeteor = mmeteor.replace("\n","")
                mmeteor = mmeteor.strip()
                lr = mmeteor.split(".")
                mmeteor = float(lr[0]+"."+lr[1][0:2])
              if line.startswith("RESULT: baseline: TER: AVG:"):
                mter = line.split(":")[4]
                mter = mter.replace("\n","")
                mter = mter.strip()
                lr = mter.split(".")
                mter = float(lr[0]+"."+lr[1][0:2])

            logger.info("Meteor = %.2f | BLEU = %.2f | TER = %.2f",
			mmeteor, mbleu, mter)

            return mmeteor, mbleu, mter

    def extract_references(self, directory, val=True):
        """
        Get reference descriptions for split we are generating outputs for.

        Helper function for bleu_score().
        """
        prefix = "val" if val else "test"
        references = self.data_gen.get_refs_by_split_as_list(prefix)

        for refid in xrange(len(references[0])):
            codecs.open('%s/%s_reference.ref%d'
                        % (directory, prefix, refid), 'w', 'utf-8').write('\n'.join([x[refid] for x in references]))

    def build_model(self, generate=False):
        '''
        Build a Keras model if one does not yet exist.

        Helper function for generate().
        '''

        if generate:
            t = self.args.generation_timesteps
        else:
            t = self.data_gen.max_seq_len
        if self.args.mrnn:
            m = models.MRNN(self.args.embed_size, self.args.hidden_size,
                            self.vocab_len,
                            self.args.dropin,
                            self.args.optimiser, self.args.l2reg,
                            hsn_size=self.hsn_size,
                            weights=self.args.checkpoint,
                            gru=self.args.gru,
                            clipnorm=self.args.clipnorm,
                            t=t)
        else:
            m = models.NIC(self.args.embed_size, self.args.hidden_size,
                           self.vocab_len,
                           self.args.dropin,
                           self.args.optimiser, self.args.l2reg,
                           hsn_size=self.hsn_size,
                           weights=self.args.checkpoint,
                           gru=self.args.gru,
                           clipnorm=self.args.clipnorm,
                           t=t)

        self.model = m.buildKerasModel(use_sourcelang=self.use_sourcelang,
                                       use_image=self.use_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate descriptions from a trained model")

    # General options
    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")
    parser.add_argument("--enable_val_pplx", action="store_true",
                        default=True,
                        help="Calculate and report smoothed validation pplx\
                        alongside the Keras objective function loss.\
                        (default=true)")
    parser.add_argument("--fixed_seed", action="store_true",
                        help="Start with a fixed random seed? Useful for\
                        reproding experiments. (default = False)")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")
    parser.add_argument("--model_checkpoints", type=str, required=True,
                        help="Path to the checkpointed parameters")
    parser.add_argument("--best_pplx", action="store_true",
                        help="Use the best PPLX checkpoint instead of the\
                        best BLEU checkpoint? Default = False.")

    # Define the types of input data the model will receive
    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")
    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=3", default=3)
    parser.add_argument("--existing_vocab", type=str, default="",
                        help="Use an existing vocabulary model to define the\
                        vocabulary and UNKing in this dataset?\
                        (default = "", which means we will derive the\
                        vocabulary from the training dataset")
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")
    parser.add_argument("--source_vectors", default=None, type=str,
                        help="Path to final hidden representations of\
                        encoder/source language VisualWordLSTM model.\
                        (default: None.) Expects a final_hidden_representation\
                        vector for each image in the dataset")
    parser.add_argument("--source_enc", type=str, default=None,
                        help="Which type of source encoder features? Expects\
                        either 'mt_enc' or 'vis_enc'. Required.")
    parser.add_argument("--source_type", type=str, default=None,
                        help="Source features over gold or predicted tokens?\
                        Expects 'gold' or 'predicted'. Required")
    parser.add_argument("--source_merge", type=str, default="sum",
                        help="How to merge source features. Only applies if \
                        there are multiple feature vectors. Expects 'sum', \
                        'avg', or 'concat'.")

    # Model hyperparameters
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--embed_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                        of LSTM recurrent state? (default = False)")
    parser.add_argument("--big_batch_size", default=10000, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 loads entire dataset. Default is 10000")
    parser.add_argument("--mrnn", action="store_true",
                        help="Use a Mao-style multimodal recurrent neural\
                        network?")
    parser.add_argument("--peeking_source", action="store_true",
                        help="Input the source features at every timestep?\
                        Default=False.")

    # Optimisation details
    parser.add_argument("--optimiser", default="adam", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--beta1", default=None, type=float)
    parser.add_argument("--beta2", default=None, type=float)
    parser.add_argument("--epsilon", default=None, type=float)
    parser.add_argument("--stopping_loss", default="bleu", type=str,
                        help="minimise cross-entropy or maximise BLEU?")
    parser.add_argument("--l2reg", default=1e-8, type=float,
                        help="L2 cost penalty. Default=1e-8")
    parser.add_argument("--clipnorm", default=-1, type=float,
                        help="Clip gradients? (default = -1, which means\
                        don't clip the gradients.")
    parser.add_argument("--max_epochs", default=50, type=int,
                        help="Maxmimum number of training epochs. Used with\
                        --predefined_epochs")
    parser.add_argument("--patience", type=int, default=10, help="Training\
                        will be terminated if validation BLEU score does not\
                        increase for this number of epochs")
    parser.add_argument("--no_early_stopping", action="store_true")

    # Language generation details
    parser.add_argument("--generation_timesteps", default=10, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")
    parser.add_argument("--test", action="store_true",
                        help="Generate for the test images? (Default=False)\
                        which means we will generate for the val images")
    parser.add_argument("--without_scores", action="store_true",
                        help="Don't calculate BLEU or perplexity. Useful if\
                        you only want to see the generated sentences or if\
                        you don't have ground-truth sentences for evaluation.")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Number of hypotheses to consider when decoding.\
                        Default=1, which means arg max decoding.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output while decoding? If you choose\
                        verbose output then you'll see the total beam search\
                        decoding process. (Default = False)")
    parser.add_argument("--multeval", action="store_true",
                        help="Evaluate using multeval?")
    parser.add_argument("--no_pplx", action="store_true",
                        help="Skip perplexity calculation?")
    parser.add_argument("--keep_file", type=str,
                        help="Path to JSON file with the words to keep. Should\
                        be a dict with keys WHOLEWORD, ENDSWITH, STARTSWITH.")
    parser.add_argument("--maximum_length", type=int, default=50,
                        help="Maximum length of sequences permissible\
			in the training data (Default = 50)")

    # Legacy options
    parser.add_argument("--generate_from_N_words", type=int, default=0,
                        help="Use N words as starting point when generating\
                        strings. Useful mostly for mt-only model (in other\
                        cases, image provides enough useful starting\
                        context.)")
    parser.add_argument("--predefined_epochs", action="store_true",
                        help="Do you want to stop training after a specified\
                        number of epochs, regardless of early-stopping\
                        criteria? Use in conjunction with --max_epochs.")

    # Neccesary but unused in this module
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")
    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")

    w = GroundedTranslationGenerator(parser.parse_args())
    w.generate()
