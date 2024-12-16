import keras
import keras.backend as K
import tensorflow as tf
from metaqnn.attack.utils import perform_attacks_per_key
from sca_utils import rankKey
import numpy as np
from ciphers.sca import AesSca

@keras.saving.register_keras_serializable()
class KeyRankMetric(keras.metrics.Metric):
    def __init__(self, name='key_rank', **kwargs):
        super(KeyRankMetric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(
            name='acc_sum', shape=(256, 256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):        
        delta = tf_calculate_key_prob(y_true, y_pred)
        delta = tf.ensure_shape(delta, [256, 256])
        self.acc_sum.assign_add(delta)
        
    def result(self):
        return tf.numpy_function(rk_key, [self.acc_sum], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256, 256))
        
@tf.function
def tf_calculate_key_prob(y_true, y_pred):
    _ret = tf.numpy_function(calculate_key_prob, [y_true, y_pred], tf.float32)
    return _ret

def calculate_key_prob(y_true, y_pred):
    ret = np.zeros((256, 256), dtype=np.float32)
    
    meta = y_true[:, -3:]
    if meta[0][0]==1:
        plaintext = meta[:, 1:2].astype(np.uint8)
        key = meta[:, 2:].astype(np.uint8)

        key_probabilities = perform_attacks_per_key(
            y_pred, plaintext, key, target_byte=0, cipher=AesSca, leakage="sbox",
            n_attacks=1, output_rank=False)
        
        key_values = np.arange(0, 256)
        i=0
        for k in key_values:
            filter = key[:, 0] == k
            if np.sum(filter) > 0:
                ret[k, :] = key_probabilities[i]
                i += 1
    return ret

# Objective: GE
def rk_key(key_probabilities):
    ranks = []
    for atk_key_byte, proba in enumerate(key_probabilities):
        if np.sum(proba)!=0:
            rank_ak = rankKey(proba, atk_key_byte)
            ranks.append(rank_ak)
    ge = np.mean(ranks) if ranks else 256
    return ge
    
