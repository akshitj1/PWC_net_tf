from datetime import datetime as dt
from sintel_dataset import get_sintel_disparity_dataset, get_sintel_path
from model import build_model
import tensorflow as tf
import platform
from model import nearest_multiple, pyramid_compatible_shape

print(tf.__version__)

def train(colab_env):
    EPOCHS = 50
    BATCH_SIZE = 8
    DATASET_SIZE = 194
    STEPS_PER_EPOCH = DATASET_SIZE/BATCH_SIZE
    NUM_PYRAMID_LEVELS = 8
    PREDICT_LEVEL = 2
    DATASET_SHAPE = (436, 1024) # h, w
    MODEL_IN_SHAPE = pyramid_compatible_shape(DATASET_SHAPE, NUM_PYRAMID_LEVELS)

    sintel_tfrecord_path = get_sintel_path(colab_env)
    train_dataset = get_sintel_disparity_dataset(sintel_tfrecord_path, BATCH_SIZE, MODEL_IN_SHAPE, NUM_PYRAMID_LEVELS)
    model = build_model(MODEL_IN_SHAPE, NUM_PYRAMID_LEVELS, PREDICT_LEVEL)

    # metrics logging
    log_dir = "logs/fit/" + dt.now().strftime("%m%d-%H%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch')

    # checkpointing
    ckpt_path = '/content/drive/My Drive/sintel_dataset/pwc_net_ckpts/ckpt' if colab_env else 'pwc_net_ckpts/ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='disparity_accuracy',
        mode='max')
    
    model.fit(train_dataset, callbacks=[
              tensorboard_callback, model_checkpoint_callback], epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    # model.save_weights(ckpt_path)


if __name__ == "__main__":
    train(colab_env=(platform.system() == 'Linux'))
