import tensorflow as tf
import numpy as np


def build_toy_model(image_shape):
    l_view = tf.keras.Input(
        shape=image_shape, dtype=tf.dtypes.int32, name='{}_image_input'.format('left_view'))
    r_view = tf.keras.Input(
        shape=image_shape, dtype=tf.dtypes.int32, name='{}_image_input'.format('right_view'))

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./255, name='image_rescale_unit')

    l_view_normed = rescale(l_view)
    r_view_normed = rescale(r_view)

    x = tf.keras.layers.Concatenate(axis=3)([l_view_normed, r_view_normed])
    x = tf.keras.layers.Conv2D(1, 3, padding='same')(x)
    disparity = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    model = tf.keras.Model(inputs={
        'left_view': l_view, 'right_view': r_view}, outputs=disparity, name='Disp_net_toy')
    # todo: replace with pyramid loss
    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        0.001), loss='mae')

    return model
