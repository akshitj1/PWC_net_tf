import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
import numpy as np
from tensorflow import keras as K

def print_n_ret(x):
    tf.print(x)
    return x

def kprint(x):
    return layers.Lambda(print_n_ret)(x)

def get_encoder():
    encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
    x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    encoder_output = layers.GlobalMaxPooling2D()(x)
    encoder_output = kprint(encoder_output)
    return keras.Model(encoder_input, encoder_output, name="encoder")


def get_decoder():
    decoder_input = keras.Input(shape=(16,), name="encoded_img")
    x = layers.Reshape((4, 4, 1))(decoder_input)
    x = kprint(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
    x = layers.UpSampling2D(3)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
    decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)
    return keras.Model(decoder_input, decoder_output, name="decoder")

def get_autoencoder():
    autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
    encoded_img = get_encoder()(autoencoder_input)
    decoded_img = get_decoder()(encoded_img)
    autoencoder = keras.Model(autoencoder_input, decoded_img, name="autoencoder")
    autoencoder.compile(loss='mean_squared_error')
    return autoencoder

def compute(x):
    return get_autoencoder()(x)

if __name__ == "__main__":
    batch_data = np.ones((1,28,28,1))#tf.ones((1,28,28,1), dtype=tf.float32)
    compute(batch_data)

    # tensorboard_callback = callbacks.TensorBoard(log_dir='toy_log', histogram_freq=1, update_freq='batch')