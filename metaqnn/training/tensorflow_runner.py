from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List
import numpy as np

from metaqnn.grammar.state_enumerator import State
from metaqnn.attack import utils
from metaqnn.training.one_cycle_lr import OneCycleLR
from metaqnn.data_loader import loadDataset_pyDataset, loadDataset_tf
import tensorflow as tf
from keras.optimizers import Adam
import keras
import gc


class TensorFlowRunner(object):
    def __init__(self, state_space_parameters, hyper_parameters):
        self.ssp = state_space_parameters
        self.hp = hyper_parameters
        self.key = self.hp.KEY
        self.cipher = utils.getCipher(self.hp.CIPHER)    
    

    @staticmethod
    def compile_model(state_list: List[State], loss, metric_list):
        
        _optimizer = Adam()  # Learning rate will be handled by OneCycleLR policy
        if len(state_list) < 1:
            raise Exception("Illegal neural net")  # TODO create clearer/better exception (class)

        model = keras.Sequential()
        for state in state_list:
            model.add(state.to_tensorflow_layer())
        model.compile(optimizer=_optimizer, loss=loss, metrics=metric_list, run_eagerly=True)
        return model

    @staticmethod
    def clear_session():
        #for gpu in strategy.extended.worker_devices:
        #    print(f"Clearing session on {gpu}")
        #    # Use tf.device(gpu) to ensure correct context placement
        #    with tf.device(gpu):
        #    # Clear the session on the current worker only
        #        keras.utils.clear_session(free_memory=True)
        keras.backend.clear_session()
        gc.collect()
        print("Session cleared")


    @staticmethod
    def count_trainable_params(model):
        return np.sum([np.prod(w.shape) for w in model.trainable_variables])

    @staticmethod
    def get_strategy():
        return tf.distribute.MirroredStrategy()

    def train_and_predict(self, model, iteration, parallel_no=1):
        # create dataloaders 
        train_db, valid_db, test_db = loadDataset_pyDataset(
                                                        self.hp.DB_FILE, key_byte=0, 
                                                        leakage=self.hp.ATTACK_LEAKAGE,
                                                        batch_size=self.hp.TRAIN_BATCH_SIZE)
        
        model.fit(
            x=train_db,
            epochs=self.hp.MAX_EPOCHS,
            validation_data=valid_db,
            shuffle=False,
            callbacks=[
                OneCycleLR(
                    max_lr=self.hp.MAX_LR * parallel_no, batch_size=self.hp.TRAIN_BATCH_SIZE * parallel_no, 
                    end_percentage=0.2, scale_percentage=0.1,
                    samples=self.hp.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                    maximum_momentum=None,
                    minimum_momentum=None, 
                    verbose=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=f"{self.hp.TRAINED_MODEL_DIR}/{self.hp.MODEL_NAME}_{iteration:04}.keras",
                    monitor='val_key_rank',
                    verbose=0,
                    mode='min',
                    save_best_only=True
                ),
                keras.callbacks.EarlyStopping(
                    monitor='val_key_rank',
                    patience=5,
                    start_from_epoch=10,
                    mode='min',
                    restore_best_weights=True,
                    verbose=True,
                    baseline=127
                )
            ]
        )
   
        return (
            model.predict(test_db),
            model.evaluate(x=test_db)
        )

    def perform_attacks_parallel(self,
                                 predictions,
                                 save_graph: bool = False,
                                 filename: str = None,
                                 folder: str = None) -> np.ndarray:
        
        ranks = utils.perform_attacks_per_key(predictions, self.hp.PTEXTS, self.hp.KEY,
                                              self.hp.ATTACK_KEY_BYTE,  self.hp.NUM_ATTACKS, self.cipher,
                                              self.hp.ATTACK_LEAKAGE)
        ge = np.mean(ranks, axis=0)-1
        
        if save_graph:
            utils.plot_ge(ge, self.hp.TRACES_PER_ATTACK, self.hp.NUM_ATTACKS, filename=filename, folder=folder)
            
        return ge

