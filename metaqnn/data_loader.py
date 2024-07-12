import numpy as np

import os
import numpy as np
import tensorflow as tf

import keras
from keras.utils import to_categorical, PyDataset

class ClassifierDataset(PyDataset):
    def __init__(self, data_dir, target_byte=0, which_subset='train', batch_size=256, shuffle=True, **kwargs):
        
        super().__init__(**kwargs)
        
        self.windows = np.load(
            os.path.join(data_dir, f'{which_subset}_windows.npy'), mmap_mode='r')

        self.target = np.load(
            os.path.join(data_dir, f'{which_subset}_targets.npy'), mmap_mode='r')
        
        self.shuffle = shuffle
        self.target_byte = target_byte

        self.which_subset = which_subset
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        # Compute the number of batches.  
        num_batches = np.ceil(self.target.shape[0] / self.batch_size).astype(int)

        return num_batches
    
    def __getitem__(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, self.target.shape[0])
        idx = self.indexes[low:high]
        
        x = self.windows[idx]
        
        # Normalize x (mean 0, std 1)
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        
        y = self.target[idx, self.target_byte]
        
        y = to_categorical(y, num_classes=256)
        
        return keras.ops.cast(x, dtype="float32"), y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.target.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)