import tensorflow as tf
import pathlib
import cv2 as cv
import numpy as np


def get_dataset_shape(im_paths):
    # most common image shape of dataset
    # df = pd.DataFrame(im_paths, columns=['fname'])
    # df['shape'] = df.fname.apply(lambda p: _read_image(p).shape)
    # im_shape = df.groupby('shape').count().idxmax().fname
    im_shape = (375, 1242, 3)
    return im_shape


def list_files(dir):
    dir_path = pathlib.Path(dir)
    assert(dir_path.is_dir())
    # ds contains depth only for _10 images
    file_paths = sorted(dir_path.glob('[!.]*_10.png'))
    print('{} contains {} files'.format(dir, len(file_paths)))
    return file_paths


def _read_image(im_path, des_shape, grayscale=False):
    # todo: do we need to adjust disparity with resize?
    if grayscale:
        im = cv.imread(str(im_path), flags=cv.IMREAD_GRAYSCALE)
    else:
        im = cv.imread(str(im_path))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return cv.resize(im, (des_shape[1], des_shape[0])).astype(np.int32)


def _read_data_paths_entry(path_rows, im_shape):
    for l_path, r_path, d_path in path_rows:
        feats = {
            'left_view': _read_image(l_path, im_shape),
            'right_view': _read_image(r_path, im_shape)
        }
        disparity = _read_image(d_path, im_shape, grayscale=True)
        yield (feats, disparity)


def _read_data_types_and_shapes(path_rows, im_shape):
    """Gets dtypes and shapes for all keys in the dataset."""
    elements = _read_data_paths_entry(path_rows, im_shape)
    feats, disparity = next(elements)
    elements.close()

    x_dtypes = {k: tf.as_dtype(v.dtype) for k, v in feats.items()}
    y_dtype = tf.as_dtype(disparity.dtype)
    dtypes = (x_dtypes, y_dtype)

    x_shapes = {k: v.shape for k, v in feats.items()}
    y_shapes = disparity.shape
    shapes = (x_shapes, y_shapes)

    return dtypes, shapes


def get_kitti_stereo_dataset(data_dir, training=True):
    # l: image_2, r: image_3, d: disp_occ_0
    train_dir = '{}/training'.format(data_dir)
    test_dir = '{}/testing'.format(data_dir)
    data_dir = train_dir if training else test_dir
    img_left_dir = '{}/image_2'.format(data_dir)
    img_right_dir = '{}/image_3'.format(data_dir)
    depth_dir = '{}/disp_occ_0'.format(data_dir)

    path_rows = zip(list_files(img_left_dir), list_files(
        img_right_dir), list_files(depth_dir))
    # todo: shuffle rows
    # left view shape of feats(x)
    img_shape = get_dataset_shape(None)  # shapes[0]['left_view']
    print('image shape: ', img_shape)

    dtypes, shapes = _read_data_types_and_shapes(path_rows, img_shape)

    dataset = tf.data.Dataset.from_generator(
        lambda: _read_data_paths_entry(path_rows, img_shape), output_types=dtypes, output_shapes=shapes)

    dataset = dataset.batch(2).prefetch(tf.data.experimental.AUTOTUNE)
    # todo: optimize io

    return dataset, img_shape
