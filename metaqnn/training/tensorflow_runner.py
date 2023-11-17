from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np

from metaqnn.grammar.state_enumerator import State
from metaqnn.attack import utils
from metaqnn.training.one_cycle_lr import OneCycleLR
import metaqnn.data_loader as data_loader
import tensorflow as tf
from keras.optimizers import Adam
from keras import backend as K


class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.key = self.hp.KEY
        self.precomputed_byte_values = self.hp.ATTACK_PRECOMPUTED_BYTE_VALUES        
    

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list):
        
        _optimizer = Adam()  # Learning rate will be handled by OneCycleLR policy
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

        model = tf.keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, loss=loss, metrics=metric_list)
        return model

    @staticmethod
    def clear_session():
        K.clear_session()

    @staticmethod
    def count_trainable_params(model):
        return np.sum([K.count_params(w) for w in model.trainable_weights])

    @staticmethod
    def get_strategy():
        return tf.distribute.MirroredStrategy()

    def train_and_predict(self, model, parallel_no=1):
        # create dataloaders
        self.train_db = data_loader.ClassifierDataset(self.hp.DB_FILE, 'train', self.hp.TRAIN_BATCH_SIZE)
        self.valid_db = data_loader.ClassifierDataset(self.hp.DB_FILE, 'valid', self.hp.TRAIN_BATCH_SIZE, shuffle=False)
        self.test_db = data_loader.ClassifierDataset(self.hp.DB_FILE, 'test', self.hp.TRAIN_BATCH_SIZE, shuffle=False)
        
        model.fit(
            x = self.train_db,
            epochs=self.hp.MAX_EPOCHS,
            validation_data = self.valid_db,
            shuffle = False,
            callbacks=[
                OneCycleLR(
                    max_lr=self.hp.MAX_LR * parallel_no, batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no, samples=self.hp.INPUT_SIZE, end_percentage=0.2, scale_percentage=0.1,
                    maximum_momentum=None,
                    minimum_momentum=None, verbose=True
                )
            ]
        )

        return (
            model.predict(self.test_db),
            model.evaluate(x=self.valid_db)
        )

    def perform_attacks(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=False, save_graph=save_graph, filename=filename, folder=folder
        )

    def perform_attacks_parallel(self, predictions, save_graph: bool = False, filename: str = None, folder: str = None):
        return utils.perform_attacks_precomputed_byte_n_parallel(
            self.hp.TRACES_PER_ATTACK, predictions, self.hp.NUM_ATTACKS, self.precomputed_byte_values, self.key,
            self.hp.ATTACK_KEY_BYTE, shuffle=False, save_graph=save_graph, filename=filename, folder=folder
        )

