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
def perform_attacks_per_key(predictions, plaintexts, true_keys, target_byte,
                    n_attacks=None, cipher: CipherSca = AesSca(), leakage='sbox',
                    output_rank=True) -> np.ndarray:

    # If all the true keys are the same
    if np.all(true_keys[:, target_byte] == true_keys[0, target_byte]):
        ranks = _attackSingleKey(
            predictions, plaintexts, true_keys, target_byte, cipher, leakage, n_attacks,
            output_rank)
    else:
        ranks = _attackVariousKey(
            predictions, plaintexts, true_keys, target_byte, cipher, leakage,
            output_rank)

    return ranks


def _attackSingleKey(predictions, plaintexts, true_keys, target_byte, cipher, leakage, n_attacks,
                     output_rank):
    '''
    Perform the attack when all the true keys are the same.
    Repeat the attack `n_attacks` times, on different batch of predictions.
    '''
    ranks = []
    for chunk in tqdm(range(n_attacks), desc='Performing attacks', leave=False):
        keys_filtered = true_keys[chunk::n_attacks]
        predictions_filtered = predictions[chunk::n_attacks]
        plains_filtered = plaintexts[chunk::n_attacks]

        key_probabilities = sortPredictions(
            predictions_filtered, plains_filtered, target_byte, cipher, leakage)

        if output_rank:
            atk_key_byte = cipher.attackedKeyByte(keys_filtered[0], target_byte)
            rank_ak, _ = guessMetrics(np.log(key_probabilities), atk_key_byte)
            ranks.append(rank_ak)
        else:
            ranks.append(np.sum(np.log(key_probabilities), axis=0))
    return ranks


def _attackVariousKey(predictions, plaintexts, true_keys, target_byte, cipher, leakage,
                      output_rank):
    '''
    Perform the attack when the true keys are different.
    Repeat the attack for each possible key value.
    '''
    ranks = []
    key_values = np.arange(0, 256)

    for k in tqdm(key_values, desc='Performing attacks', leave=False):
        filter = true_keys[:, target_byte] == k

        if np.sum(filter) > 0:
            keys_filtered = true_keys[filter]
            predictions_filtered = predictions[filter]
            plains_filtered = plaintexts[filter]

            key_probabilities = sortPredictions(
                predictions_filtered, plains_filtered, target_byte, cipher, leakage)

            if output_rank:
                atk_key_byte = cipher.attackedKeyByte(keys_filtered[0], target_byte)
                rank_ak, _ = guessMetrics(np.log(key_probabilities), atk_key_byte)
                ranks.append(rank_ak)
            else:
                ranks.append(np.sum(np.log(key_probabilities), axis=0))
            
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
        
def convertLeakage(leakage: str):
    if 'hw' in leakage.lower():
        return 'hw'
    else:
        return 'identity'
