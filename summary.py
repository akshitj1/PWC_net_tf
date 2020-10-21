import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import io
# https://www.tensorflow.org/tensorboard/image_summaries

def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_disparities(feats, disps_true, disps_pred, num_pyr_levels):
    fig, ax = plt.subplots(nrows=(num_pyr_levels+2)//2, ncols=4, figsize=(20.0, 10.0))
    l_img = feats['left_view'].squeeze()
    r_img = feats['right_view'].squeeze()
    grid = lambda idx: ax[np.unravel_index(idx, ax.shape)]
    grid(0).imshow(l_img)
    grid(0).set_title('left view')
    grid(1).imshow(r_img)
    grid(1).set_title('right view')

    ax_idx=2
    for lvl in range(num_pyr_levels):
        l_key='l{}'.format(lvl)
        disp_true = disps_true[l_key].squeeze()
        disp_pred = disps_pred[l_key].squeeze()
        grid(ax_idx).imshow(disp_true)
        grid(ax_idx).set_title('level {}: disparity gt'.format(lvl))
        ax_idx+=1
        grid(ax_idx).imshow(disp_pred)
        grid(ax_idx).set_title('level {}: disparity pred'.format(lvl))
        ax_idx+=1
    grid(ax_idx).set_visible(False)
    grid(ax_idx+1).set_visible(False)

    fig.tight_layout()
    return fig
