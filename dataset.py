import tensorflow as tf
import pathlib
import numpy as np
import platform


# tfrecord ref:
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://stackoverflow.com/questions/45427637/numpy-to-tfrecords-is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrec/45428167#45428167


def list_files(dir):
    dir_path = pathlib.Path(dir)
    assert(dir_path.is_dir())
    # ds contains depth only for _10 images
    file_paths = sorted(dir_path.glob('[!.]*_10.png'))
    print('{} contains {} files'.format(dir, len(file_paths)))
    return file_paths


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def all_eq(l):
    return l.count(l[0]) == len(l)


def get_stereo_fts_example(l_path, r_path, d_path):
    l_im_str, r_im_str, d_im_str = tuple(
        map(lambda p: open(p, 'rb').read(), [l_path, r_path, d_path]))
    seq_ids = [pathlib.Path(p).stem for p in [l_path, r_path, d_path]]
    assert all_eq(seq_ids)
    seq_id = bytes(seq_ids[0], 'ascii')
    feature = {
        'sequence_id': _bytes_feature(seq_id),
        'left_view_raw': _bytes_feature(l_im_str),
        'right_view_raw': _bytes_feature(r_im_str),
        'depth_raw': _bytes_feature(d_im_str),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def generate_kitti_stereo_tfrecord_dataset(data_dir, out_record_file, training=True):
    # l: image_2, r: image_3, d: disp_occ_0
    train_dir = '{}/training'.format(data_dir)
    test_dir = '{}/testing'.format(data_dir)
    data_dir = train_dir if training else test_dir
    img_left_dir = '{}/colored_0'.format(data_dir)
    img_right_dir = '{}/colored_1'.format(data_dir)
    depth_dir = '{}/flow_occ'.format(data_dir)

    path_rows = zip(list_files(img_left_dir), list_files(
        img_right_dir), list_files(depth_dir))

    with tf.io.TFRecordWriter(out_record_file) as writer:
        for l_path, r_path, d_path in path_rows:
            tf_example = get_stereo_fts_example(l_path, r_path, d_path)
            writer.write(tf_example.SerializeToString())


def _parse_ds_entry(entry_proto, record_description):
    return tf.io.parse_single_example(entry_proto, record_description)


def decode_image(im_raw, grayscale=False):
    channels = 1 if grayscale else 3
    im = tf.io.decode_image(contents=im_raw, channels=channels,
                            expand_animations=False, name='decode_png_bytes_to_tensor')
    return im


def byte_features_to_tensors(ft_entry):
    feats = {
        'left_view': decode_image(ft_entry['left_view_raw']),
        'right_view': decode_image(ft_entry['right_view_raw']),
        'depth': decode_image(ft_entry['depth_raw'], True)
    }
    return feats


def adapt_to_model_input(ft_entry, out_shape):
    h, w, _ = out_shape
    resized_fts = {}
    for key, im in ft_entry.items():
        # todo: do we need to adjust disparity with resize?
        # cv.resize(v, (w, h)).astype(np.int32)
        resized_fts[key] = tf.image.resize(
            im, tf.constant([h, w]), name='uniform_sizing')
    feats = dict((k, resized_fts[k]) for k in ('left_view', 'right_view'))
    disparity = resized_fts['depth']
    return (feats, disparity)


def get_kitti_stereo_dataset(stereo_records_file):
    raw_stereo_dataset = tf.data.TFRecordDataset(stereo_records_file)

    # Create a dictionary describing the features.
    stereo_record_description = {
        'sequence_id': tf.io.FixedLenFeature([], tf.string),
        'left_view_raw': tf.io.FixedLenFeature([], tf.string),
        'right_view_raw': tf.io.FixedLenFeature([], tf.string),
        'depth_raw': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_stereo_dataset = raw_stereo_dataset.map(
        lambda entry: _parse_ds_entry(entry, stereo_record_description))

    np_fts_dataset = parsed_stereo_dataset.map(byte_features_to_tensors)

    # computed by get_dataset_stats
    model_in_shape = (376, 1241, 3)
    print('image shape: ', model_in_shape)
    uniformly_shaped_dataset = np_fts_dataset.map(
        lambda entry: adapt_to_model_input(entry, model_in_shape))

    dataset = uniformly_shaped_dataset.repeat().batch(
        2).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, model_in_shape


if __name__ == "__main__":
    colab_env = platform.system() == 'Linux'
    gdrive_kitti_dir = '/content/drive/My Drive/kitti_dataset/stereo_disp'
    local_kitti_dir = '/Users/akshitjain/ext/workspace/datasets/kitti_2012/stereo_flow'
    out_tfrecord_path = 'data/kitti_2012_stereo_flow.tfrecords'
    kitti_data_dir = gdrive_kitti_dir if colab_env else local_kitti_dir
    generate_kitti_stereo_tfrecord_dataset(local_kitti_dir, out_tfrecord_path)
