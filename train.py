from datetime import datetime as dt
from dataset import get_kitti_stereo_dataset
from model import build_model
import os
import tensorflow as tf
import platform

print(tf.__version__)


def get_tpu_strategy():
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    strategy = tf.distribute.TPUStrategy(resolver)
    return strategy


def train(use_tpu=True):
    kitti_data_dir = '/content/drive/My Drive/kitti_dataset/stereo_disp' if use_tpu else '/Users/akshitjain/ext/workspace/datasets/kitti_2012/stereo_flow'
    train_dataset, img_shape = get_kitti_stereo_dataset(kitti_data_dir)

    if use_tpu:
        tpu_run_strategy = get_tpu_strategy()
        with tpu_run_strategy.scope():
            model = build_model(img_shape)
    else:
        model = build_model(img_shape)

    log_dir = "logs/fit/" + dt.now().strftime("%m%d-%H%M")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch')
    model.fit(train_dataset, callbacks=[tensorboard_callback])


if __name__ == "__main__":
    train(use_tpu=(platform.system() == 'Linux'))
