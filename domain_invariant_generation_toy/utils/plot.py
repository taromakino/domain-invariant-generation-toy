import numpy as np


def plot_red_green_image(ax, image):
    '''
    Input image has shape (2, m, n)
    '''
    _, m, n = image.shape
    image = image.transpose((1, 2, 0))
    image = np.dstack((image, np.zeros((m, n))))
    ax.imshow(image)