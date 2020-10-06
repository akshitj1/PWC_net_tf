from datetime import datetime as dt
from dataset import get_kitti_stereo_dataset
from model import build_model
import os
import tensorflow as tf
import platform

print(tf.__version__)


def train(colab_env):
    gdrive_kitti_dir = '/content/drive/My Drive/kitti_dataset/stereo_disp'
    local_kitti_dir = '/Users/akshitjain/ext/workspace/datasets/kitti_2012/stereo_flow'
    kitti_data_dir = gdrive_kitti_dir if colab_env else local_kitti_dir
    train_dataset, img_shape = get_kitti_stereo_dataset(kitti_data_dir)
    model = build_model(img_shape)

    log_dir = "logs/fit/" + dt.now().strftime("%m%d-%H%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch')
    model.fit(train_dataset, callbacks=[tensorboard_callback])
    model.save_weights('ckpt')


if __name__ == "__main__":
    train(colab_env=(platform.system() == 'Linux'))
