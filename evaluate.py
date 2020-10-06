import tensorflow as tf
from dataset import get_kitti_stereo_dataset
from model import build_model


def eval():
    kitti_dir = '/Users/akshitjain/ext/workspace/datasets/kitti_2012/stereo_flow'
    test_ds, im_shape = get_kitti_stereo_dataset(
        kitti_dir, True)
    model = build_model(im_shape)
    model.load_weights('ckpt')
    disparity = model.predict(test_ds)
    print(disparity)


if __name__ == "__main__":
    eval()
