import tensorflow as tf

print(tf.__version__)


def conv(output_channels=16, kernel_size=3, downsample=False):
    # todo: add relu
    stride = 2 if downsample else 1
    return tf.keras.layers.Conv2D(filters=output_channels,
                                  kernel_size=kernel_size, strides=stride)


def deconv(output_channels=2, kernel_size=4, upsample=True):
    stride = 2 if upsample else 1
    # todo: add padding. bias is true for deconv? check default in pytroch vs tf
    return tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                           kernel_size=kernel_size, strides=stride)


def downsample_conv_seq(output_channels=16):
    conv_a = conv(output_channels, downsample=True)
    conv_aa = conv(output_channels)
    conv_b = conv(output_channels)
    return tf.keras.Sequential([conv_a, conv_aa, conv_b])


def extract_features_multiscale(num_layers=6):
    layers = []
    input_channels = 3
    output_channels = 16
    for layer_idx in range(1, num_layers+1):
        layers.append(downsample_conv_seq(output_channels))
        output_channels *= 2
    return layers


def build_image_input(image_width=100, image_height=100, color_channels=3):
    return tf.keras.Input(shape=(image_height, image_width, color_channels),
                          batch_size=1, dtype=tf.dtypes.uint8)


def correlation_volume(im_features_l, im_features_r_warped, max_disp_x=50):
    # input B*H*W*F, B*return H*W*D
    corr_slices = []
    for disp_x in range(max_disp_x):
        # shift with zero padding instead of roll
        im_r_shifted = tf.roll(im_features_r_warped, shift=-disp_x, axis=2)
        corr_slice = tf.reduce_mean(tf.multiply(
            im_features_l, im_r_shifted), axis=-1)
        corr_slices.append(corr_slice)
    return tf.stack(corr_slices)


def corr_to_flow(corr_vol, num_conv_reps=5):
    # reduce channels from dispartiy to 2 by aplying convolutions repeatedly
    res_channels = [128, 96, 64, 32, 2]
    assert len(res_channels) == num_conv_reps
    conv_corr_vol = corr_vol
    for i in range(num_conv_reps):
        conv_corr_vol = tf.concat(
            conv_corr_vol, conv(res_channels[0])(conv_corr_vol))
    # last layer is 2 channel, hence flow
    pred_flow = conv_corr_vol
    return pred_flow


def warp(im_features, flow):
    return im_features


def build_model():
    LEFT, RIGHT = (0, 1)
    img = (build_image_input(), build_image_input())
    img_features = tuple(map(extract_features_multiscale, img))

    prev_flow = None
    for level in range(7, 0, -1):
        if prev_flow is None:
            flow_up = tf.zeros(tf.shape(img_features[RIGHT][level])[:-1])
        else:
            flow_up = deconv()(prev_flow)
        # todo: flow is scaled originally?
        img_r_warped = warp(img_features[RIGHT][level], flow_up)
        img_l_features = img_features[LEFT][level]
        corr_vol = correlation_volume(img_l_features, img_r_warped)
        # todo: upsampled features convolved to 2 channels also concatenated in original paper
        # left features are concatenated to create U-Net like architecture
        corr_vol_enriched = tf.concat(
            [corr_vol, img_l_features, flow_up], axis=1)

        flow = corr_to_flow(corr_vol, corr_vol_enriched)
        prev_flow = flow
