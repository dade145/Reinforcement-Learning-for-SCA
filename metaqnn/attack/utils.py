from sca_utils import sortPredictions, guessMetrics
from ciphers.sca import *
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import numpy as np
from matplotlib import pyplot as plot
import multiprocessing as mp
import random
import os
from os import path
from typing import Callable

import matplotlib
matplotlib.use('Agg')


CPU_COUNT = (len(os.sched_getaffinity(0))
             if 'sched_getaffinity' in dir(os) else mp.cpu_count())

def plot_ge(rk_avg, traces_per_attack, attack_amount, filename='fig', folder='data'):
    plot.rcParams['figure.figsize'] = (20, 10)
    plot.ylim(-5, 180)
    plot.xlim(0, traces_per_attack + 1)
    plot.grid(True)
    plot.plot(range(1, traces_per_attack + 1), rk_avg, '-')
    plot.xlabel('Number of traces')
    plot.ylabel('Mean rank of correct key guess')

    plot.title(
        f'{filename} Guessing Entropy\nUp to {traces_per_attack:d} traces averaged over {attack_amount:d} attacks',
        loc='center'
    )

    plot.savefig(
        path.normpath(path.join(
            folder, f'{filename}_{traces_per_attack:d}trs_{attack_amount:d}att.svg')),
        format='svg', dpi=1200, bbox_inches='tight'
    )
    plot.close()


###############
#  FUNCTIONS  #
###############

# Performs attack
def perform_attacks_per_key(predictions, ptexts, keys, target_byte, n_attacks, cipher: CipherSca):
    ranks = []
    key_values = np.random.choice(np.arange(0, 256), n_attacks, replace=False)
    
    for k in tqdm(key_values, desc='Performing attacks', leave=False):
        key_filter = keys[:, 0] == k
        if np.sum(key_filter) > 0:
            keys_perKey = keys[key_filter]
            predictions_perKey = predictions[key_filter]
            plains_perKey = ptexts[key_filter]

            mapping = [cipher.invAttackedIntermediate(plains_perKey, np.array(
                [i]*len(plains_perKey)), target_byte) for i in range(256)]
            key_proba = sortPredictions(predictions_perKey, np.array(mapping).T)
            atk_key_byte = cipher.attackedKeyByte(keys_perKey[0], target_byte)
            rank_ak, _ = guessMetrics(np.log(key_proba), atk_key_byte)
            ranks.append(rank_ak - 1)
    return ranks


def getCipher(cipher_name: str):
    if cipher_name.upper() == 'AES':
        return AesSca()
    elif cipher_name.upper() == 'CLEFIA':
        return ClefiaSca()
    elif cipher_name.upper() == 'CAMELLIA':
        return CamelliaSca()
    elif cipher_name.upper() == 'SEED':
        return SeedSca()
    else:
        raise ValueError(
            f'Unknown cipher {cipher_name}. Choose between: aes, clefia, camellia, or seed.')
