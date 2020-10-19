import tensorflow as tf
from pathlib import Path
import numpy as np
import platform

# tfrecord ref:
# https://www.tensorflow.org/tutorials/load_data/tfrecord
# https://stackoverflow.com/questions/45427637/numpy-to-tfrecords-is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrec/45428167#45428167


def get_frame_ids(view_dir):
    # lets get metadata first
    view_dir = Path(view_dir)
    assert(view_dir.is_dir())
    scene_dirs = list(view_dir.glob('[!.]*'))
    scene_names = [p.stem for p in scene_dirs]
    frames = []
    for scene_dir, scene_name in zip(scene_dirs, scene_names):
        frame_paths = list(scene_dir.glob('[!.]*.png'))
        frame_names = [p.stem for p in frame_paths]
        frames.extend(['{}/{}'.format(scene_name, fname) for fname in frame_names])
    print('total {} egs. found'.format(len(frames)))
    return frames

def scene_frame_to_path(sintel_base_dir,view_sub_dir, fr_name):
    return Path('{}/{}/{}.png'.format(sintel_base_dir, view_sub_dir, fr_name))

def get_data_paths(sintel_base_dir):
    frame_names = get_frame_ids('{}/clean_left'.format(sintel_base_dir))
    sintel_dataset =[]
    for fname in frame_names:
        entry = {
            'frame_name': fname,
            'left_view': scene_frame_to_path(sintel_base_dir, 'clean_left', fname),
            'right_view': scene_frame_to_path(sintel_base_dir, 'clean_right', fname),
            'depth': scene_frame_to_path(sintel_base_dir, 'disparities', fname),
            'occlusions' : scene_frame_to_path(sintel_base_dir, 'occlusions', fname)
        }
        sintel_dataset.append(entry)
    return sintel_dataset

def generate_sintel_disparity_tfrecord_dataset(data_dir, out_record_file, training=True):
    # l: image_2, r: image_3, d: disp_occ_0
    train_dir = '{}/training'.format(data_dir)
    test_dir = '{}/testing'.format(data_dir)
    data_dir = train_dir if training else test_dir
    entry_paths = get_data_paths(data_dir)

    with tf.io.TFRecordWriter(out_record_file) as writer:
        for e in entry_paths:
            tf_example = get_stereo_fts_example(e['frame_name'],e['left_view'], e['right_view'], e['depth'])
            writer.write(tf_example.SerializeToString())



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_stereo_fts_example(frame_id, l_path, r_path, d_path):
    l_im_str, r_im_str, d_im_str = tuple(
        map(lambda p: open(p, 'rb').read(), [l_path, r_path, d_path]))
    seq_id = bytes(frame_id, 'ascii')
    feature = {
        'sequence_id': _bytes_feature(seq_id),
        'left_view_raw': _bytes_feature(l_im_str),
        'right_view_raw': _bytes_feature(r_im_str),
        'depth_raw': _bytes_feature(d_im_str),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _parse_ds_entry(entry_proto, record_description):
    return tf.io.parse_single_example(entry_proto, record_description)


def decode_image(im_raw, grayscale=False):
    channels = 1 if grayscale else 3
    im = tf.io.decode_image(contents=im_raw, channels=channels,
                            expand_animations=False, name='decode_png_bytes_to_tensor')
    return im

def sintel_rgb_to_depth(encoded_depth_im):
    # input is uint8 image with 3 channels 2^2 * r + g/2^6 + b/2^14
    # output is float32 depth. depth varies from 0 to 1024
    # details in sintel I/O scripts
    im = tf.cast(encoded_depth_im, tf.float32)
    depth =  tf.reduce_sum(tf.multiply(tf.constant([4., 1./2**6, 1./2**14]), im), axis=-1, keepdims=True)
    return depth

def byte_features_to_tensors(ft_entry):
    feats = {
        'left_view': decode_image(ft_entry['left_view_raw']),
        'right_view': decode_image(ft_entry['right_view_raw']),
        'depth': sintel_rgb_to_depth(decode_image(ft_entry['depth_raw']))
    }
    return feats

def pad(im, out_shape):
    h, w = out_shape
    return tf.image.resize_with_crop_or_pad(im, h, w)


def adapt_to_model_input(ft_entry, out_shape, num_pyr_levels):
    h, w = out_shape
    resized_fts = {}
    for key, im in ft_entry.items():
        # todo: do we need to adjust disparity with resize?
        # nearest interpolation preserves dtype
        resized_fts[key] = pad(im, (h, w))
        #tf.image.resize(im, tf.constant([h, w]), method='nearest',name='uniform_sizing')
    feats = dict((k, resized_fts[k]) for k in ('left_view', 'right_view'))
    disparity = resized_fts['depth']
    # out shape is prefect multiple of 2^(num_pyr_levels-1)
    disparity_pyramid = [tf.image.resize(disparity, [h//2**lvl, w//2**lvl],method='bilinear') for lvl in range(num_pyr_levels)]
    y = dict(('l{}'.format(lvl), disparity_pyramid[lvl]) for lvl in range(num_pyr_levels))
    return (feats, y)

def get_sintel_disparity_dataset(stereo_records_file, batch_size, model_in_shape, model_pyr_levels):
    print('model input shape: ', model_in_shape)
    raw_stereo_dataset = tf.data.TFRecordDataset(stereo_records_file)

    # Create a dictionary describing the features.
    stereo_record_description = {
        'sequence_id': tf.io.FixedLenFeature([], tf.string),
        'left_view_raw': tf.io.FixedLenFeature([], tf.string),
        'right_view_raw': tf.io.FixedLenFeature([], tf.string),
        'depth_raw': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_stereo_dataset = raw_stereo_dataset.map(lambda entry: _parse_ds_entry(entry, stereo_record_description), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # cache here as flows get pyramided later
    np_fts_dataset = parsed_stereo_dataset.map(byte_features_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()    
    uniformly_shaped_dataset = np_fts_dataset.map(lambda entry: adapt_to_model_input(entry, model_in_shape, model_pyr_levels), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = uniformly_shaped_dataset.repeat().batch(
        batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

def get_sintel_path(colab_env):
    local_records = 'data/sintel_stereo_disparity.tfrecords'
    gdrive_records = '/content/drive/My Drive/sintel_dataset/sintel_stereo_disparity.tfrecords'
    records_path = gdrive_records if colab_env else local_records
    return records_path


if __name__ == "__main__":
    colab_env = (platform.system() == 'Linux')

    gdrive_sintel_dir = '/content/drive/My Drive/sintel_dataset'
    local_sintel_dir = '/Users/akshitjain/ext/workspace/datasets/sintel'
    sintel_data_dir = gdrive_sintel_dir if colab_env else local_sintel_dir

    out_tfrecord_path = get_sintel_path(colab_env)
    generate_sintel_disparity_tfrecord_dataset(sintel_data_dir, out_tfrecord_path)
