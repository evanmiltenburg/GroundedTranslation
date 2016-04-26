import glob
import matplotlib.pyplot as plt
import seaborn

plt.switch_backend('agg')

gen_logs = glob.glob('../logs/flickr30k/gen*.log')
mbs_logs = glob.glob('../logs/flickr30k/mbs*.log')

def get_BLEU(filename):
    "Get the BLEU score from the log file."
    with open(filename) as f:
        bleu_line = f.readlines()[-1]
    return float(bleu_line[7:12])

def get_beam_size(path):
    "Get the beam size from the filename."
    filename = path.split('/')[-1]
    return int(filename[4:-4])

def create_data_list(logs):
    "Turn a list of filenames into a list of tuples."
    return zip(*sorted((get_beam_size(log),get_BLEU(log)) for log in logs))

gen_x, gen_y = create_data_list(gen_logs)
mbs_x, mbs_y = create_data_list(mbs_logs)

gen_line, = plt.plot(gen_x, gen_y, '-o', label='Beam search')
mbs_line, = plt.plot(mbs_x, mbs_y, '-o', label='Dual beam search')

plt.legend(handles=[gen_line, mbs_line])
plt.xlabel('Beam size')
plt.ylabel('BLEU score')

plt.savefig('beam_vs_bleu.pdf')
