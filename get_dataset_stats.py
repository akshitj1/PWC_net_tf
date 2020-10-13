import cv2 as cv
import pathlib
import pandas as pd


def _read_image(im_path):
    return cv.imread(str(im_path))


def list_files(dir):
    dir_path = pathlib.Path(dir)
    assert(dir_path.is_dir())
    # ds contains depth only for _10 images
    file_paths = sorted(dir_path.glob('[!.]*_10.png'))
    print('{} contains {} files'.format(dir, len(file_paths)))
    return file_paths


def get_dataset_shape(im_paths):
    # most common image shape of dataset
    df = pd.DataFrame(im_paths, columns=['fname'])
    df['shape'] = df.fname.apply(lambda p: _read_image(p).shape)
    im_shape = df.groupby('shape').count().idxmax().fname
    return im_shape


if __name__ == "__main__":
    im_shape = get_dataset_shape(list_files(
        '/Users/akshitjain/ext/workspace/datasets/kitti_2012/stereo_flow/training/colored_0'))
    print(im_shape)
