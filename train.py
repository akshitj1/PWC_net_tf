from datetime import datetime as dt
from dataset import get_kitti_stereo_dataset
from model import build_model
import os
import tensorflow as tf
import platform

print(tf.__version__)


def train(colab_env):
    EPOCHS = 10
    BATCH_SIZE = 2
    DATASET_SIZE = 194
    STEPS_PER_EPOCH = DATASET_SIZE/BATCH_SIZE

    kitti_records_path = 'data/kitti_2012_stereo_flow.tfrecords'
    train_dataset, img_shape = get_kitti_stereo_dataset(kitti_records_path)
    model = build_model(img_shape)

    # metrics logging
    log_dir = "logs/fit/" + dt.now().strftime("%m%d-%H%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch')

    # checkpointing
    ckpt_path = '/content/drive/My Drive/kitti_dataset/pwc_net_ckpts/ckpt' if colab_env else 'pwc_net_ckpts/ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True)
    model.fit(train_dataset, callbacks=[
              tensorboard_callback, model_checkpoint_callback], epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)
    model.save_weights(ckpt_path)


if __name__ == "__main__":
    train(colab_env=(platform.system() == 'Linux'))
