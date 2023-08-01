import numpy as np


def hist_discrete(ax, x):
    n_bins = len(np.unique(x))
    ax.hist(x, bins=n_bins)


def plot_grayscale_image(ax, image):
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)


def plot_red_green_image(ax, image):
    '''
    Input image has shape (2, m, n)
    '''
    _, m, n = image.shape
    image = image.transpose((1, 2, 0))
    image = np.dstack((image, np.zeros((m, n))))
    ax.imshow(image)