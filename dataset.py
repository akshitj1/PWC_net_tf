import tensorflow as tf
import pathlib
from PIL import Image
import numpy as np


def list_files(dir):
    dir_path = pathlib.Path(dir)
    assert(dir_path.is_dir())
    file_paths = sorted(dir_path.glob('[!.]*.png'))
    return file_paths


def _read_image(im_path):
    return np.asarray(Image.open(im_path), dtype=np.int32)


def _read_data_paths_entry(path_rows):
    for l_path, r_path, d_path in path_rows:
        feats = {
            'left_view': _read_image(l_path),
            'right_view': _read_image(r_path)
        }
        disparity = _read_image(d_path)
        yield (feats, disparity)


def _read_data_types_and_shapes(path_rows):
    """Gets dtypes and shapes for all keys in the dataset."""
    elements = _read_data_paths_entry(path_rows)
    feats, disparity = next(elements)
    elements.close()

    x_dtypes = {k: tf.as_dtype(v.dtype) for k, v in feats.items()}
    y_dtype = tf.as_dtype(disparity.dtype)
    dtypes = (x_dtypes, y_dtype)

    x_shapes = {k: v.shape for k, v in feats.items()}
    y_shapes = disparity.shape
    shapes = (x_shapes, y_shapes)

    return dtypes, shapes


def get_kitti_stereo_dataset(data_dir):
    # l: image_2, r: image_3, d: disp_occ_0
    train_dir = '{}/training'.format(data_dir)
    img_left_dir = '{}/image_2'.format(train_dir)
    img_right_dir = '{}/image_3'.format(train_dir)
    depth_dir = '{}/disp_occ_0'.format(train_dir)

    path_rows = zip(list_files(img_left_dir), list_files(
        img_right_dir), list_files(depth_dir))
    # todo: shuffle rows

    dtypes, shapes = _read_data_types_and_shapes(path_rows)

    dataset = tf.data.Dataset.from_generator(
        lambda: _read_data_paths_entry(path_rows), output_types=dtypes, output_shapes=shapes)
    dataset = dataset.batch(1)
    # todo: optimize io for gpu processing

    # left view shape of feats(x)
    img_shape = shapes[0]['left_view']  # [1:]
    print('image shape: ', img_shape)
    return dataset, img_shape
