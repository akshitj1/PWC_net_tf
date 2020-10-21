from datetime import datetime as dt
from sintel_dataset import get_sintel_disparity_dataset, get_sintel_path, get_sintel_raw_path,generate_sintel_disparity_tfrecord_dataset
from model import build_model
from pathlib import Path
import tensorflow as tf
from tensorflow import keras as K
import platform
from model import nearest_multiple, pyramid_compatible_shape
from summary import plot_disparities, plot_to_image, plot_disparity_histograms

print(tf.__version__)

def train(colab_env):
    EPOCHS = 225
    BATCH_SIZE = 16
    DATASET_SIZE = 1064
    STEPS_PER_EPOCH = DATASET_SIZE/BATCH_SIZE
    NUM_PYRAMID_LEVELS = 8
    PREDICT_LEVEL = 2
    DATASET_SHAPE = (436, 1024) # h, w
    MODEL_IN_SHAPE = pyramid_compatible_shape(DATASET_SHAPE, NUM_PYRAMID_LEVELS)

    sintel_tfrecord_path = get_sintel_path(colab_env)
    if not Path(sintel_tfrecord_path).exists():
        sintel_data_dir = get_sintel_raw_path(colab_env)
        print("WARNING: Dataset tfRecord file: {} not found. Creating one from directory: ".format(sintel_tfrecord_path, sintel_data_dir))
        generate_sintel_disparity_tfrecord_dataset(sintel_data_dir, sintel_tfrecord_path)

    train_dataset, test_dataset = get_sintel_disparity_dataset(sintel_tfrecord_path, BATCH_SIZE, MODEL_IN_SHAPE, NUM_PYRAMID_LEVELS)
    model = build_model(MODEL_IN_SHAPE, NUM_PYRAMID_LEVELS, PREDICT_LEVEL)
    
    # metrics logging
    log_base_dir = '/content/drive/My Drive/sintel_dataset/pwc_net_logs' if colab_env else 'logs'
    log_dir = "{}/fit/{}".format(log_base_dir ,dt.now().strftime("%m%d-%H%M"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1, update_freq='batch')

    file_writer = tf.summary.create_file_writer(log_dir+'/disp_pyramid')

    def log_disparity_pyramid(epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_batch = list(test_dataset.take(1).as_numpy_iterator())
        feats, disps_true = test_batch[0]
        disps_pred = model.predict(test_batch)

        fig = plot_disparities(feats, disps_true, disps_pred, NUM_PYRAMID_LEVELS)
        im_disp = plot_to_image(fig)
        fig = plot_disparity_histograms(disps_true, disps_pred, NUM_PYRAMID_LEVELS)
        im_hist = plot_to_image(fig)
        
        with file_writer.as_default():
            tf.summary.image("Disparity pyramid", im_disp, step=epoch)
            tf.summary.image("Disparity Distribution", im_hist, step=epoch)


    # Define the per-epoch callback.
    disparity_plot_callback = K.callbacks.LambdaCallback(on_epoch_end=log_disparity_pyramid)


    # checkpointing
    ckpt_path = '/content/drive/My Drive/sintel_dataset/pwc_net_ckpts/ckpt' if colab_env else 'pwc_net_ckpts/ckpt'

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='tf_op_layer_l0_disparity_accuracy',
        mode='max')
    

    # train at 1e-4 lr for 2/3rd of epochs
    first_phase_epochs = (2*EPOCHS)//3
    model.fit(train_dataset,
        callbacks=[
              tensorboard_callback, 
              model_checkpoint_callback, 
              disparity_plot_callback],
        epochs=first_phase_epochs, 
        steps_per_epoch=STEPS_PER_EPOCH)
    print('first phase of learning complete. changing learning rate.')

    #https://stackoverflow.com/questions/59737875/keras-change-learning-rate
    K.backend.set_value(model.optimizer.learning_rate, 5e-5)
    remaining_epochs = EPOCHS-first_phase_epochs
    model.fit(train_dataset,
        callbacks=[
              tensorboard_callback, 
              model_checkpoint_callback, 
              disparity_plot_callback],
        initial_epoch=first_phase_epochs,
        epochs=remaining_epochs,
        steps_per_epoch=STEPS_PER_EPOCH)
    print('training complete!!!')


if __name__ == "__main__":
    train(colab_env=(platform.system() == 'Linux'))
