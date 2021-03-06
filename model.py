import tensorflow as tf
from tensorflow import keras as K
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
import tensorflow_addons as tfa
import numpy as np

# shared modules using keras functional API: https://github.com/google-research/google-research/blob/master/video_structure/vision.py


def conv(output_channels, kernel_size=3, dilation_rate=1, downsample=False, name=None):
    stride = 2 if downsample else 1
    conv2d = tf.keras.layers.Conv2D(filters=output_channels,
                                    kernel_size=kernel_size,
                                    strides=stride,
                                    padding='same',
                                    dilation_rate=dilation_rate,
                                    kernel_regularizer=K.regularizers.l2(4e-4),
                                    bias_regularizer=K.regularizers.l2(4e-4),
                                    name='{}_conv'.format(name) if name else None)
    relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    return tf.keras.Sequential([conv2d, relu])


def deconv(output_channels=2, kernel_size=4, upsample=True, name=None):
    stride = 2 if upsample else 1
    return tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           padding='same',
                                           kernel_regularizer=K.regularizers.l2(4e-4),
                                           bias_regularizer=K.regularizers.l2(4e-4),
                                           name='{}_deconv'.format(name))


def downsample_block(output_channels=16, name=None):
    conv_a = conv(output_channels, downsample=True, name='downsample_2')
    conv_aa = conv(output_channels, name='upreceptive_1')
    conv_b = conv(output_channels, name='upreceptive_2')
    return tf.keras.Sequential([conv_a, conv_aa, conv_b], name)


def build_pyramid_feature_extractor(image_shape, num_levels, predict=False):
    h, w = image_shape
    # level 0 -> original resolution, level n -> l/2**n dims for l in {w,h}
    # there are 6 levels from 2 to 7. we will not store 0,1 as they are not used
    img = tf.keras.Input(shape=(h, w, 3), batch_size=(1 if predict else None), name='image_input')
    ft_pyramid = [None]*num_levels
    # 0th level features is the image itself. anyway we are going to use from level 2
    ft_pyramid[0] = img
    lvl_op_channels = [None, 8, 16, 32, 64, 96, 128, 196]
    assert(len(lvl_op_channels) == num_levels)

    for lvl in range(1, num_levels):
        ft_pyramid[lvl] = downsample_block(
            lvl_op_channels[lvl], name='level_{}'.format(lvl))(ft_pyramid[lvl-1])
    return tf.keras.Model(inputs=img, outputs=ft_pyramid, name='image_feature_extractor')


def get_image_input(image_shape, name, predict=False):
    h, w=image_shape
    batch_size = 1 if predict else None
    return tf.keras.Input(shape=(h, w, 3), batch_size=batch_size, dtype=tf.dtypes.uint8, name=name)


def nearest_multiple(dividend, divisor):
    q, r = divmod(dividend, divisor)
    return (q + (r > 0))*divisor

def pyramid_compatible_shape(in_shape, num_levels):
    divisor = 2**(num_levels-1)
    in_h, in_w = in_shape
    out_h, out_w = (nearest_multiple(in_h, divisor),
                    nearest_multiple(in_w, divisor))
    out_shape = (out_h, out_w)
    return out_shape

def pad(im, in_img_shape, num_levels):
    divisor = 2**(num_levels-1)
    in_h, in_w, in_c = in_img_shape
    out_h, out_w = (nearest_multiple(in_h, divisor),
                    nearest_multiple(in_w, divisor))
    out_img_shape = (out_h, out_w, in_c)
    return tf.image.resize_with_crop_or_pad(im, out_h, out_w), out_img_shape


def crop(im, out_shape):
    h, w, _ = out_shape
    return tf.image.resize_with_crop_or_pad(im, h, w)


def build_preprocess_image(img_shape, num_levels, predict=False):
    im = get_image_input(img_shape, predict=predict, name='preprocess_input')
    im_normed = tf.cast(im, dtype=tf.float32)/255.
    return tf.keras.Model(inputs=im, outputs=im_normed, name='preprocessing_layer')


def correlation_volume(im_features_l, im_features_r_warped, max_disp_x=4):
    # input B*H*W*F, B*return H*W*D
    corr_slices = []
    im_features_r_padded = tf.keras.layers.ZeroPadding2D(
        padding=(0, max_disp_x))(im_features_r_warped)
    # new size is w+2*d
    max_padded_disp_x = 2*max_disp_x+1

    # iterate -d to +d
    for disp_x in range(max_padded_disp_x):
        im_r_shifted = tf.slice(im_features_r_padded, [
                                0, 0, disp_x, 0], tf.shape(im_features_l))
        corr_slice = tf.reduce_mean(tf.multiply(
            im_features_l, im_r_shifted), axis=-1)
        corr_slices.append(corr_slice)
    return tf.stack(corr_slices, axis=3, name='correlation_volume')


# class CorrelationVolumeToFlow(tf.keras.layers.Layer):
#     def __init__(self, num_conv_reps=5):
#         super(Linear, self).__init__()
#         self.num_conv_reps = num_conv_reps
#         res_channels = [128, 96, 64, 32, 2]
#         assert(len(res_channels) == num_conv_reps)
#         self.rep_conv = [None]*res_channels
#         for i in range(num_conv_reps):
#             self.rep_conv[i] = conv(res_channels[i], name='rep_{}'.format(i))
#         self.contract_flow = conv(res_channels[-1],
#                                   name='compress_flow_channels')

#     def call(self, corr_vol):
#         # reduce channels from dispartiy to 2 by aplying convolutions repeatedly
#         conv_corr_vol = corr_vol
#         for i in range(self.num_conv_reps):
#             conv_corr_vol = tf.keras.layers.Concatenate(axis=3)(
#                 [conv_corr_vol, self.rep_conv[i](conv_corr_vol)])
#         # last layer is 2 channel, hence flow
#         pred_flow = self.contract_flow(conv_corr_vol)
#         return pred_flow


def predict_disp(ft, num_conv_reps=5):
    # reduce channels from dispartiy to 1 by aplying convolutions repeatedly
    res_channels = [128, 96, 64, 32, 1]
    assert(len(res_channels) == num_conv_reps)
    conv_ft = ft
    for i in range(num_conv_reps-1):
        # we donot use dense connections as of now
        conv_ft = conv(res_channels[i])(conv_ft)
    # original paper does not applies relu on last layer as they want both positive and negative flows but we want only
    # positive disparities so relu is fine.
    # last layer is 1 channel, hence disparity
    pred_disp = conv(res_channels[-1])(conv_ft)
    # last 32 dim ft. can be passed as upsampled features
    return pred_disp, conv_ft


def warp(im_features_r, disp_r_to_l__x):
    # disparity is positive
    # 0->y, 1->x
    disp_y = tf.zeros_like(disp_r_to_l__x, name='zero_disp_y')
    flow_r_to_l = tf.concat([disp_y, disp_r_to_l__x], axis=-1)
    im_rec_l = tfa.image.dense_image_warp(im_features_r, flow_r_to_l)
    return im_rec_l


def extract_multiscale_disps(img_features, num_levels, predict_level):
    LEFT, RIGHT = (0, 1)
    # disparity from r to l in L ref. frame
    disps = [None]*num_levels
    warp_layer = tf.keras.layers.Lambda(lambda x: warp(x[0], x[1]))
    for level in range(num_levels-1, predict_level-1, -1):
        im_l_fts = img_features[LEFT][level]
        im_r_fts = img_features[RIGHT][level]

        if level == num_levels-1:
            corr_vol = correlation_volume(im_l_fts, im_r_fts)
            corr_vol = tf.keras.layers.LeakyReLU(0.1)(corr_vol)
            corr_vol_enriched = tf.keras.layers.Concatenate(axis=3)(
                [corr_vol, im_l_fts])
        else:
            # upsample previous level flows and features
            disp_up = deconv(1, name='disp_upsample_{}'.format(
                level+1))(disps[level+1])
            res_feat_up = deconv(1, name='feat_upsample_{}'.format(
                level+1))(res_feat)

            # original paper scales gt flow by 20 but not in our case. hence, adjust accordingly
            # flow is computed as original res disparity irrespective the level
            upsampled_disp_scale_factor = 1./2**level
            im_l_rec = warp_layer(
                inputs=[im_r_fts, upsampled_disp_scale_factor*disp_up])
            corr_vol = correlation_volume(im_l_fts, im_l_rec)
            corr_vol = tf.keras.layers.LeakyReLU(0.1)(corr_vol)
            # left features are concatenated to create U-Net like architecture
            # disparity is added from low resolution to upscale resolution, hence final disparity will be far greater than max_disp at each level
            corr_vol_enriched = tf.keras.layers.Concatenate(axis=3)(
                [corr_vol, im_l_fts, disp_up, res_feat_up])

        disps[level], res_feat = predict_disp(corr_vol_enriched)
    return disps


def spatial_refine_flow(flow):
    channels = [128, 128, 128, 96, 64, 32]
    dil_rates = [1, 2, 4, 8, 16, 1]
    assert(len(channels) == len(dil_rates))
    disparity_delta = flow
    for channel, dil_rate in zip(channels, dil_rates):
        disparity_delta = conv(channel, 3, dilation_rate=dil_rate)(disparity_delta)
    # we are computing deltas so we want negative values too
    disparity_delta = tf.keras.layers.Conv2D(
        1, 
        3, 
        padding='same', 
        kernel_regularizer=K.regularizers.l2(4e-4),
        bias_regularizer=K.regularizers.l2(4e-4),
        name='disparity_delta')(disparity_delta)
    return disparity_delta


def masked_avg_epe_loss(disparity_true, disparity_pred):
    # input shapes are [B*H*W*1]
    # gt disparity is sparse, compute loss only on pixels where we have disparity info
    mask = disparity_true >= 1
    disparity_true = tf.cast(disparity_true, tf.float32)
    abs_error = tf.math.abs(
        tf.math.subtract(disparity_pred, disparity_true))
    aepe = tf.math.reduce_mean(
        tf.boolean_mask(abs_error, mask))
    return aepe

def epe_loss(disparity_true, disparity_pred):
    # input can be of any resolution scale of pyramid
    abs_error = tf.math.abs(
        tf.math.subtract(disparity_pred, disparity_true))
    aepe = tf.math.reduce_mean(abs_error)
    return aepe

def avg_epe_accuracy(disparity_true, disparity_pred):
    abs_error = tf.math.abs(
        tf.math.subtract(disparity_pred, disparity_true))
    aepe = tf.math.reduce_mean(abs_error)
    return aepe

class DebuggableModel(K.Model):
    # https://keras.io/examples/keras_recipes/debugging_tips/#tip-3-to-debug-what-happens-during-fit-use-runeagerlytrue
    # https://keras.io/getting_started/intro_to_keras_for_researchers/
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # For custom training steps, users can just write:
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        for var, grad in zip(trainable_variables, gradients):
            grad=grad.numpy()
            print(var.name, np.mean(np.abs(grad)))
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}


def build_model(img_shape, num_pyramid_levels, predict_level, predict=False):
    views = [{'id': 0, 'name': 'left_view'}, {'id': 1, 'name': 'right_view'}]
    LEFT, RIGHT = (0, 1)
    num_pyramid_levels = 8
    # 0-indexed, level at which prediction will happen
    predict_level = 2
    preprocess_image = build_preprocess_image(
        img_shape, num_pyramid_levels, predict=predict)
    extract_ft_pyramid = build_pyramid_feature_extractor(
        img_shape, num_pyramid_levels, predict=predict)

    imgs = []
    ims_ft_pyramid = []

    for view in views:
        im = get_image_input(img_shape, predict=predict, name='{}_image'.format(view['name']))
        imgs.append(im)
        im_normed = preprocess_image(im)
        im_ft_pyramid = extract_ft_pyramid(im_normed)
        ims_ft_pyramid.append(im_ft_pyramid)

    flow_pyramid = extract_multiscale_disps(
        ims_ft_pyramid, num_pyramid_levels, predict_level)
    
    # bilinear flow till before predict level
    for lvl in range(predict_level):
        h,w  = img_shape
        flow_pyramid[lvl] = tf.image.resize(flow_pyramid[predict_level], (h//2**lvl, w//2**lvl))
    
    if predict:
        # build loss from these flows
        model = K.Model(inputs={
                            'left_view': imgs[0], 'right_view': imgs[1]}, outputs=flow_pyramid[0], name='PWC_net')
        model.compile(loss=epe_loss)
        print(model.summary())
    else:
        # noisy_flow = flow_pyramid[predict_level]
        # smooth_flow = noisy_flow + spatial_refine_flow(noisy_flow)

        # original resolution bilinear upsample
        # h, w = img_shape
        # l0_flow = tf.image.resize(smooth_flow, (h, w))

        # output_flow = crop(l0_flow, img_shape)
        # identity to rename, as this name gets used in loss fn.
        out_flows = dict(('l{}'.format(lvl),tf.identity(flow_pyramid[lvl], 'l{}'.format(lvl))) for lvl in range(num_pyramid_levels))

        # build loss from these flows
        model = K.Model(inputs={
                            'left_view': imgs[0], 'right_view': imgs[1]}, outputs=out_flows, name='PWC_net')
        losses = dict(('l{}'.format(lvl),epe_loss) for lvl in range(num_pyramid_levels))
        # no loss on layers before predict level
        # originally flow is scaled by 20. But, in our exp. this leads to very less weight to reg. losses
        pyr_loss_weights = [0., 0., 0.32, 0.08, 0.02, 0.01, 0.005, 0.0025]
        pyr_loss_weights = dict(('l{}'.format(lvl),pyr_loss_weights[lvl]) for lvl in range(num_pyramid_levels))

        assert(len(pyr_loss_weights) == num_pyramid_levels)

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
        model.compile(optimizer=optimizer, loss=losses, loss_weights=pyr_loss_weights,
                    metrics={'l0': avg_epe_accuracy})

    return model
