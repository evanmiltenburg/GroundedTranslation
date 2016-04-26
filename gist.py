import argparse
import json

def get_parser():
    "Get an argument parser for this module."
    parser = argparse.ArgumentParser(
        description="Train an word embedding model using LSTM network")
    
    parser.add_argument("--run_string", default="", type=str,
                        help="Optional string to help you identify the run")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug messages to stdout?")
    parser.add_argument("--init_from_checkpoint", help="Initialise the model\
                        parameters from a pre-defined checkpoint? Useful to\
                        continue training a model.", default=None, type=str)
    parser.add_argument("--enable_val_pplx", action="store_true",
                        default=True,
                        help="Calculate and report smoothed validation pplx\
                        instead of Keras objective function loss. Turns off\
                        calculation of Keras val loss. (default=true)")
    parser.add_argument("--generate_from_N_words", type=int, default=0,
                        help="Use N words as starting point when generating\
                        strings. Useful mostly for mt-only model (in other\
                        cases, image provides enough useful starting\
                        context.)")
    
    parser.add_argument("--small", action="store_true",
                        help="Run on 100 images. Useful for debugging")
    parser.add_argument("--num_sents", default=5, type=int,
                        help="Number of descriptions/image for training")
    parser.add_argument("--small_val", action="store_true",
                        help="Validate on 100 images. Useful for speed/memory")
    
    # These options turn off image or source language inputs.
    # Image data is *always* included in the hdf5 dataset, even if --no_image
    # is set.
    parser.add_argument("--no_image", action="store_true",
                        help="Do not use image data.")
    # If --source_vectors = None: model uses only visual/image input, no
    # source language/encoder hidden layer representation feature vectors.
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
    
    parser.add_argument("--dataset", default="", type=str, help="Path to the\
                        HDF5 dataset to use for training / val input\
                        (defaults to flickr8k)")
    parser.add_argument("--supertrain_datasets", nargs="+", help="Paths to the\
                        datasets to use as additional training input (defaults\
                        to None)")
    
    parser.add_argument("--big_batch_size", default=10000, type=int,
                        help="Number of examples to load from disk at a time;\
                        0 loads entire dataset. Default is 10000")
    
    parser.add_argument("--predefined_epochs", action="store_true",
                        help="Do you want to stop training after a specified\
                        number of epochs, regardless of early-stopping\
                        criteria? Use in conjunction with --max_epochs.")
    parser.add_argument("--max_epochs", default=50, type=int,
                        help="Maxmimum number of training epochs. Used with\
                        --predefined_epochs")
    parser.add_argument("--patience", type=int, default=10, help="Training\
                        will be terminated if validation BLEU score does not\
                        increase for this number of epochs")
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--dropin", default=0.5, type=float,
                        help="Prob. of dropping embedding units. Default=0.5")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead\
                        of LSTM recurrent state? (default = False)")
    
    parser.add_argument("--optimiser", default="adam", type=str,
                        help="Optimiser: rmsprop, momentum, adagrad, etc.")
    parser.add_argument("--lr", default=None, type=float)
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
    
    parser.add_argument("--unk", type=int,
                        help="unknown character cut-off. Default=3", default=3)
    parser.add_argument("--generation_timesteps", default=30, type=int,
                        help="Maximum number of words to generate for unseen\
                        data (default=10).")
    parser.add_argument("--h5_writeable", action="store_true",
                        help="Open the H5 file for write-access? Useful for\
                        serialising hidden states to disk. (default = False)")
    
    parser.add_argument("--use_predicted_tokens", action="store_true",
                        help="Generate final hidden state\
                        activations over oracle inputs or from predicted\
                        inputs? Default = False ( == Oracle)")
    parser.add_argument("--fixed_seed", action="store_true",
                        help="Start with a fixed random seed? Useful for\
                        reproding experiments. (default = False)")
    parser.add_argument("--existing_vocab", type=str, default="",
                        help="Use an existing vocabulary model to define the\
                        vocabulary and UNKing in this dataset?\
                        (default = "", which means we will derive the\
                        vocabulary from the training dataset")
    parser.add_argument("--no_early_stopping", action="store_true")
    
    parser.add_argument("--mrnn", action="store_true",
                        help="Use a Mao-style multimodal recurrent neural\
                        network?")
    parser.add_argument("--seed_value", type=int, default=1234,
                        help="Provide specific seed value.")
    return parser

def json_to_arg_string(filename):
    "Loads a JSON file and converts its contents to an argument string."
    with open(filename) as f:
        d = json.load(f)
    args = []
    for argument, value in d.items():
        if value is None:
            argument = '--' + argument
            args.append(argument)
        else:
            argument = ''.join(['--', argument, '=', str(value)])
            args.append(argument)
    return args

d = dict()
d['mrnn'] = None # This gets saved as `null` in the JSON file.
d['seed_value'] = 6789
d['l2reg'] = 1e-5
with open('test.json','w') as f:
    json.dump(d,f)

parser = get_parser()
arguments = json_to_arg_string('test.json')


ns = parser.parse_args()
print 'Without arguments:'
print ns

ns = parser.parse_args(arguments)
print 'With arguments:'
print ns
