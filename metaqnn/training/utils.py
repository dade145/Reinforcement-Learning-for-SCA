import tensorflow as tf
import keras
from keras.losses import categorical_crossentropy

@keras.saving.register_keras_serializable()
def custom_loss(y_true, y_pred):
    return categorical_crossentropy(y_true[:, :-3], y_pred)

def setGpu(num_gpu):
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[num_gpu], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[num_gpu], True)
        except RuntimeError as e:
            print(e)

@keras.saving.register_keras_serializable()      
class AccuracyMetric(keras.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(AccuracyMetric, self).__init__(name=name, **kwargs)
        self.m = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.m.update_state(
            keras.ops.equal(keras.ops.argmax(y_true[:, :-3], axis=-1), keras.ops.argmax(y_pred, axis=-1)))

    def result(self):
        return self.m.result()

    def reset_states(self):
        self.m.reset_states()