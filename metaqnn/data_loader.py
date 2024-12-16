import tensorflow as tf
import numpy as np
import os
from keras.utils import to_categorical, PyDataset
from sca_utils import hw
from metaqnn.attack.utils import convertLeakage
import keras

def _loadData(data_dir, subset):
    windows_path = os.path.join(data_dir, f'{subset}_windows.npy')
    target_path = os.path.join(data_dir, f'{subset}_targets.npy')
    meta_path = os.path.join(data_dir, f'{subset}_meta.npy')
    
    windows = np.load(windows_path, mmap_mode='r')
    targets = np.load(target_path, mmap_mode='r')
    meta = np.load(meta_path, mmap_mode='r')
    
    return windows, targets, meta

def _normalize(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def _normalize_tf(x):
    return (x - tf.reduce_mean(x, axis=0)) / tf.math.reduce_std(x, axis=0)

def _getNumClasses(leakage):
    if leakage.upper() == 'HW':
        return 9
    else:
        return 256


def loadDataset_pyDataset(dataset_dir, key_byte, leakage='sbox', batch_size=256):
    leakage = convertLeakage(leakage)
    
    train_db = ClassifierDataset(dataset_dir, key_byte, leakage,
                                'train', batch_size, workers=8)
    valid_db = ClassifierDataset(dataset_dir, key_byte, leakage,
                                'valid', batch_size, shuffle=False, workers=8)
    test_db  = ClassifierDataset(dataset_dir, key_byte, leakage,
                                'test', batch_size, shuffle=False, workers=8)
    return train_db, valid_db, test_db

def loadDataset_tf(dataset_dir, key_byte, leakage='sbox', batch_size=256):
    leakage = convertLeakage(leakage)
    
    train_dataset = ClassifierDatasetTf(dataset_dir, key_byte, leakage,
                                        'train', batch_size, shuffle=True)
    valid_dataset = ClassifierDatasetTf(dataset_dir, key_byte, leakage,
                                        'valid', batch_size, shuffle=False)
    test_dataset = ClassifierDatasetTf(dataset_dir, key_byte, leakage, 
                                        'test', batch_size, shuffle=False)
    
    return train_dataset(), valid_dataset(), test_dataset()

class ClassifierDatasetTf:

    def __init__(self, data_dir, target_byte=0, leakage='identity', which_subset='train', batch_size=256, shuffle=True, **kwargs):
        
        super().__init__(**kwargs)
        assert which_subset in ['train', 'valid', 'test']
        assert leakage.upper() in ['IDENTITY', 'HW']
        
        self.windows, self.targets, self.meta = _loadData(data_dir, which_subset)
        
        self.target_byte = target_byte
        self.num_classes = _getNumClasses(leakage)

        self.which_subset = which_subset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_samples = len(self.targets)
    
    def __call__(self):
        dataset = self._createGeneratorDataset()

        dataset = dataset.map(lambda x, y: (_normalize_tf(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _shuffleIndex(self):
        self.indexes = np.arange(self.total_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def _createGeneratorDataset(self):
        
        def generator():
            self._shuffleIndex() 
            for idx in self.indexes:
                x, y, meta_info = self._getData(idx)
                y = self._to_categorical(y)
                y = self._concatenateMetaInfo(y, meta_info)
                
                yield x.astype("float32"), y

        output_signature=(
            tf.TensorSpec(shape=self.windows.shape[1:], dtype=tf.float32),
            tf.TensorSpec(shape=(self.num_classes+3,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature)

        return dataset
    
    def _getData(self, idx):
        x = self.windows[idx]
        y = self.targets[idx, self.target_byte]
        meta_info = self.meta[idx, :, self.target_byte]
        return x, y, meta_info
    
    def _to_categorical(self, y):
        if self.num_classes == '9':
            y = hw[y]
        y = to_categorical(y, self.num_classes)
        return y

    def _concatenateMetaInfo(self, y, meta_info):
        if self.which_subset == 'train':
            y = np.concatenate((y, [0], meta_info))
        else:
            y = np.concatenate((y, [1], meta_info))
        return y


class ClassifierDataset(PyDataset):
    def __init__(self, data_dir, target_byte=0, leakage='identity', which_subset='train', batch_size=256, shuffle=True, **kwargs):
        
        super().__init__(**kwargs)
        assert which_subset in ['train', 'valid', 'test']
        assert leakage.upper() in ['IDENTITY', 'HW']
        
        self.windows, self.targets, self.meta = _loadData(data_dir, which_subset)
        
        self.shuffle = shuffle
        self.target_byte = target_byte
        self.leakage = leakage.upper()

        self.which_subset = which_subset
        self.batch_size = batch_size
        self.on_epoch_end()
    
    def __len__(self):
        # Compute the number of batches.  
        num_batches = np.ceil(self.targets.shape[0] / self.batch_size).astype(int)

        return num_batches
    
    def __getitem__(self, index):
        x, y, meta_info = self._getData(index)
        x = _normalize(x)
        y = self._to_categotical(y)
        y = self._concatenateMetaInfo(y, meta_info)
        return keras.ops.cast(x, dtype="float32"), y

    def _getData(self, index):
        idx = self._getBatchIndexes(index)
        
        x = self.windows[idx]
        y = self.targets[idx, self.target_byte]
        meta = self.meta[idx, :, self.target_byte]
        return x, y, meta

    def _getBatchIndexes(self, index):
        low = index * self.batch_size
        high = min(low + self.batch_size, self.targets.shape[0])
        idx = self.indexes[low:high]
        return idx

    def _to_categotical(self, y):
        if self.leakage == 'HW':
            y = self._hammingWeight(y)
            return to_categorical(y, num_classes=9)
        else:
            return to_categorical(y, num_classes=256)
    
    def _concatenateMetaInfo(self, y, meta_info):
        meta_info = meta_info.reshape(-1, 2)
        if self.which_subset == 'train':
            return np.concatenate((y, np.zeros((len(meta_info), 1)), meta_info), axis=1).astype("float32")
        else:
            return np.concatenate((y, np.ones((len(meta_info), 1)), meta_info), axis=1).astype("float32")
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.targets.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def _hammingWeight(self, y):
        if len(np.unique(y))==9:
            return y
        else:
            return hw[y]
