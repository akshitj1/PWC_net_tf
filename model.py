import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# shared modules using keras functional API: https://github.com/google-research/google-research/blob/master/video_structure/vision.py


def conv(output_channels=16, kernel_size=3, downsample=False, name=None):
    stride = 2 if downsample else 1
    conv2d = tf.keras.layers.Conv2D(filters=output_channels,
                                    kernel_size=kernel_size, strides=stride, padding='same', name='{}_conv'.format(name) if name else None)
    relu = tf.keras.layers.LeakyReLU(alpha=0.1)
    return tf.keras.Sequential([conv2d, relu])


def deconv(output_channels=2, kernel_size=4, upsample=True, name=None):
    stride = 2 if upsample else 1
    return tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                           kernel_size=kernel_size, strides=stride, padding='same', name='{}_deconv'.format(name))


def build_downsample_block(output_channels=16, name=None):
    conv_a = conv(output_channels, downsample=True, name='downsample_2')
    conv_aa = conv(output_channels, name='upreceptive_1')
    conv_b = conv(output_channels, name='upreceptive_2')
    return tf.keras.Sequential([conv_a, conv_aa, conv_b], name)


def build_pyramid_feature_extractor(image_shape, num_levels):
    img = tf.keras.Input(shape=image_shape, name='image_input')
    output_channels = 16
    ft_lvls = [None]*num_levels
    for lvl in range(0, num_levels):
        downsample_conv_seq = build_downsample_block(
            output_channels, name='level_{}'.format(lvl))
        ft_lvl = downsample_conv_seq(
            ft_lvls[lvl-1] if lvl > 0 else img)
        ft_lvls[lvl] = ft_lvl
        output_channels *= 2
    return tf.keras.Model(inputs=img, outputs=ft_lvls, name='image_feature_extractor')


def build_image_input(image_shape, name='left'):
    return tf.keras.Input(shape=image_shape, dtype=tf.dtypes.int32, name='{}_image_input'.format(name))


def nearest_multiple(dividend, divisor):
    q, r = divmod(dividend, divisor)
    return (q + (r > 0))*divisor


def pad(img_shape, num_pyr_levels):
    im_y, im_x, channels = img_shape
    im_y_padded, im_x_padded = (nearest_multiple(
        im_y, 2**num_pyr_levels), nearest_multiple(im_x, 2**num_pyr_levels))
    paddings = ((0, im_y_padded-im_y), (0, im_x_padded-im_x))
    padded_img_shape = (im_y_padded, im_x_padded, channels)
    print('original shape: {}, padded shape: {}'.format(
        img_shape, padded_img_shape))
    return tf.keras.layers.ZeroPadding2D(paddings), padded_img_shape


def crop(in_shape, padded_shape):
    crops = np.array(padded_shape) - np.array(in_shape)
    return tf.keras.layers.Cropping2D(((0, crops[0]), (0, crops[1])))


def build_preprocess_image(img_shape, num_pyr_levels):
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(
        1./255, name='image_rescale_unit')
    # pad to ensure dims are multiple of 2**pyr_levels
    padding_layer, new_img_shape = pad(img_shape, num_pyr_levels)
    return tf.keras.models.Sequential([rescale, padding_layer], name='preprocess'), new_img_shape


def correlation_volume(im_features_l, im_features_r_warped, max_disp_x=50):
    # input B*H*W*F, B*return H*W*D
    corr_slices = []
    for disp_x in range(max_disp_x):
        # todo: shift with zero padding instead of roll
        im_r_shifted = tf.roll(im_features_r_warped, shift=-disp_x, axis=2)
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


def corr_to_disp(corr_vol, num_conv_reps=5):
    # reduce channels from dispartiy to 1 by aplying convolutions repeatedly
    res_channels = [128, 96, 64, 32, 1]
    assert(len(res_channels) == num_conv_reps)
    conv_corr_vol = corr_vol
    for i in range(num_conv_reps):
        conv_corr_vol = tf.keras.layers.Concatenate(axis=3)(
            [conv_corr_vol, conv(res_channels[i])(conv_corr_vol)])
    # last layer is 1 channel, hence disparity
    pred_disp = conv(res_channels[-1])(conv_corr_vol)
    return pred_disp


def warp(im_features_r, disp_r_to_l__x):
    # disparity is positive
    # 0->y, 1->x
    disp_y = tf.zeros(tf.shape(disp_r_to_l__x), name='zero_disp_y')
    flow_r_to_l = tf.concat([disp_y, disp_r_to_l__x], axis=3)
    im_rec_l = tfa.image.dense_image_warp(im_features_r, flow_r_to_l)
    return im_rec_l


def extract_multiscale_disps(img_features, num_scale_levels):
    LEFT, RIGHT = (0, 1)
    disps = [None]*num_scale_levels
    warp_layer = tf.keras.layers.Lambda(lambda x: warp(x[0], x[1]))
    for level in range(num_scale_levels-1, -1, -1):
        if level+1 >= num_scale_levels:
            disp_up = tf.zeros(
                tf.concat([tf.shape(img_features[RIGHT][level])[:-1], tf.constant([1], dtype=tf.int32)], axis=0), name='no_flow')
        else:
            disp_up = deconv(1, name='disp_upsample_{}'.format(
                level+1))(disps[level+1])

        # original paper scales gt flow by 20 but not in our case. hence, adjust accordingly
        # flow is computed as original res disparity irrespective the level
        upsampled_disp_scale_factor = 1./(2**(level+1+1))
        img_r_warped = warp_layer(
            inputs=[img_features[RIGHT][level], upsampled_disp_scale_factor*disp_up])
        img_l_features = img_features[LEFT][level]
        corr_vol = correlation_volume(img_l_features, img_r_warped)
        corr_vol = tf.keras.layers.LeakyReLU(0.1)(corr_vol)
        # todo: upsampled features convolved to 2 channels also concatenated in original paper
        # left features are concatenated to create U-Net like architecture
        corr_vol_enriched = tf.keras.layers.Concatenate(axis=3)(
            [corr_vol, img_l_features, disp_up])

        disps[level] = corr_to_disp(corr_vol_enriched)
    return disps


def pyramid_l1_loss(flows_pred, flow_gt):
    num_feature_scale_levels = 6
    flows_gt = []
    lvl_loss_weights = [0.0, 0.005, 0.01, 0.02, 0.08, 0.32]
    assert(len(lvl_loss_weights) == num_feature_scale_levels)
    loss = 0
    for lvl in range(num_feature_scale_levels):
        downscaled_flow = tf.image.resize(flow_gt, tf.shape(
            flows_pred[lvl])[-2:], method=tf.image.ResizeMethod.AREA)
        loss += lvl_loss_weights[lvl] * tf.reduce_sum(
            tf.abs(tf.subtract(flows_pred[lvl], downscaled_flow)))
    return loss


def masked_loss(disparity_true, disparity_pred):
    # gt disparity is sparse, compute loss only on pixels where we have disparity info
    abs_error = tf.keras.backend.abs(
        tf.subtract(disparity_pred, tf.cast(disparity_true, tf.float32)))
    mask = disparity_true > 0
    mae = tf.reduce_mean(tf.boolean_mask(abs_error, mask))
    return mae


def disparity_accuracy(disparity_true, disparity_pred):
    mask = disparity_true > 0
    disparity_true = tf.cast(disparity_true, tf.float32)
    disparity_true = disparity_true[mask]
    disparity_pred = disparity_pred[mask]
    disp_errors = tf.abs(disparity_pred-disparity_true)
    disp_errors_rel = disp_errors*100./disparity_true
    accurate_predictions = tf.reduce_sum(
        tf.cast(disp_errors_rel < 5., tf.float32))
    accuracy = accurate_predictions/tf.reduce_sum(tf.cast(mask, tf.float32))
    return accuracy


def build_model(img_shape):
    views = [{'id': 0, 'name': 'left_view'}, {'id': 1, 'name': 'right_view'}]
    LEFT, RIGHT = (0, 1)
    num_feature_scale_levels = 6

    preprocess_image, processed_img_shape = build_preprocess_image(
        img_shape, num_feature_scale_levels)
    extract_features_multiscale = build_pyramid_feature_extractor(
        processed_img_shape, num_feature_scale_levels)

    imgs = []
    ims_ft_pyramid = []

    for view in views:
        im = build_image_input(img_shape, name=view['name'])
        imgs.append(im)
        im_normed = preprocess_image(im)
        im_ft_pyramid = extract_features_multiscale(im_normed)
        ims_ft_pyramid.append(im_ft_pyramid)

    flow_pyramid = extract_multiscale_disps(
        ims_ft_pyramid, num_feature_scale_levels)

    # original resolution
    output_flow = 0.5*deconv(1, name='bilinear_upsample')(flow_pyramid[0])

    output_flow = crop(img_shape, processed_img_shape)(output_flow)
    # build loss from these flows
    model = tf.keras.Model(inputs={
                           'left_view': imgs[0], 'right_view': imgs[1]}, outputs=output_flow, name='PWC_net')
    # todo: replace with pyramid loss
    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        0.001), loss=masked_loss, metrics=[disparity_accuracy])

    return model
