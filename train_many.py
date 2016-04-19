import random
import subprocess

for i in range(5):
    new_seed = str(random.randrange(1234,5678))
    args = ['THEANO_FLAGS=floatX=float32,device=gpu1',
            'python train.py',
            '--dataset=/data/flickr30k',
            '--hidden_size=512'
            '--run_string=final_seeded_' + new_seed,
            '--unk=5',
            '--l2reg=1e-05',
            '--clipnorm=1.0',
            '--generation_timesteps=15',
            '--optimiser=adam',
            '--fixed_seed',
            '--seed_value=' + new_seed,
            '&> whathappened' + new_seed + '.log &']
    subprocess.call(' '.join(args))
