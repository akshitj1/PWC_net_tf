import tensorflow as tf
import tensorflow_addons as tfa
print(tf.__version__)


def conv(output_channels=16, kernel_size=3, downsample=False):
    # todo: add relu
    stride = 2 if downsample else 1
    return tf.keras.layers.Conv2D(filters=output_channels,
                                  kernel_size=kernel_size, strides=stride, padding='same')


def deconv(output_channels=2, kernel_size=4, upsample=True):
    stride = 2 if upsample else 1
    # todo: add padding. bias is true for deconv? check default in pytroch vs tf
    return tf.keras.layers.Conv2DTranspose(filters=output_channels,
                                           kernel_size=kernel_size, strides=stride, padding='same')


def downsample_conv_seq(x, output_channels=16):
    conv_a = conv(output_channels, downsample=True)
    conv_aa = conv(output_channels)
    conv_b = conv(output_channels)
    return conv_b(conv_aa(conv_a(x)))
    # return tf.keras.Sequential([conv_a, conv_aa, conv_b])


def extract_features_multiscale(img, num_levels=6):
    output_channels = 16
    ft_lvls = [None]*num_levels
    for lvl in range(0, num_levels):
        ft_lvl = downsample_conv_seq(
            ft_lvls[lvl-1] if lvl > 0 else img, output_channels)
        ft_lvls[lvl] = ft_lvl
        output_channels *= 2
    return ft_lvls


def build_image_input(image_width, image_height, color_channels=3):
    # todo: normalize image
    return tf.keras.Input(shape=(image_height, image_width, color_channels),
                          batch_size=1, dtype=tf.dtypes.uint8)


def preprocess_image(img):
    return tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(img)


def correlation_volume(im_features_l, im_features_r_warped, max_disp_x=50):
    # input B*H*W*F, B*return H*W*D
    corr_slices = []
    for disp_x in range(max_disp_x):
        # shift with zero padding instead of roll
        im_r_shifted = tf.roll(im_features_r_warped, shift=-disp_x, axis=2)
        corr_slice = tf.reduce_mean(tf.multiply(
            im_features_l, im_r_shifted), axis=-1)
        corr_slices.append(corr_slice)
    return tf.stack(corr_slices, axis=3)


def corr_to_flow(corr_vol, num_conv_reps=5):
    # reduce channels from dispartiy to 2 by aplying convolutions repeatedly
    res_channels = [128, 96, 64, 32, 2]
    assert(len(res_channels) == num_conv_reps)
    conv_corr_vol = corr_vol
    for i in range(num_conv_reps):
        conv_corr_vol = tf.concat(
            [conv_corr_vol, conv(res_channels[i])(conv_corr_vol)], axis=3)
    # last layer is 2 channel, hence flow
    pred_flow = conv_corr_vol
    return pred_flow


def warp(im_features, flow):
    return tfa.image.dense_image_warp(im_features, flow)


def extract_multiscale_flows(img_features, num_scale_levels):
    LEFT, RIGHT = (0, 1)
    flows = [None]*num_scale_levels
    for level in range(num_scale_levels-1, -1, -1):
        if level+1 >= num_scale_levels:
            # flow_up = tf.zeros(tf.shape(img_features[RIGHT][level])[:-1])
            flow_up = tf.zeros(
                tf.concat([tf.shape(img_features[RIGHT][level])[:-1], tf.constant([2], dtype=tf.int32)], axis=0))
        else:
            flow_up = deconv()(flows[level+1])
        # todo: flow is scaled originally?
        img_r_warped = warp(img_features[RIGHT][level], flow_up)
        img_l_features = img_features[LEFT][level]
        corr_vol = correlation_volume(img_l_features, img_r_warped)
        # todo: upsampled features convolved to 2 channels also concatenated in original paper
        # left features are concatenated to create U-Net like architecture
        corr_vol_enriched = tf.concat(
            [corr_vol, img_l_features, flow_up], axis=3)

        flow = corr_to_flow(corr_vol_enriched)
        flows[level] = flow
    return flows


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


def build_model():
    LEFT, RIGHT = (0, 1)
    num_feature_scale_levels = 6
    im_width, im_height = (1024, 1024)
    imgs = [build_image_input(im_width, im_height)
            for img_idx in [LEFT, RIGHT]]
    imgs_normed = list(map(preprocess_image, imgs))
    imgs_features = [extract_features_multiscale(
        im, num_feature_scale_levels) for im in imgs_normed]
    flows = extract_multiscale_flows(imgs_features, num_feature_scale_levels)
    # build loss from these flows
    model = tf.keras.Model(inputs=imgs, outputs=flows)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(
        0.001), loss=pyramid_l1_loss)
    return model


def train():
    tf.compat.v1.disable_eager_execution()
    # tf.config.run_functions_eagerly(True)
    model = build_model()
    print(model.summary())


if __name__ == "__main__":
    train()
