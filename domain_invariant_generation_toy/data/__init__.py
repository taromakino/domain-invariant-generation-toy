from data.colored_mnist import make_data
from data.dsprites import make_data
from utils.plot import plot_grayscale_image, plot_red_green_image


MAKE_DATA = {
    'colored_mnist': colored_mnist.make_data,
    'dsprites': dsprites.make_data
}


PLOT = {
    'colored_mnist': plot_red_green_image,
    'dsprites': plot_grayscale_image
}


IMAGE_SHAPE = {
    'colored_mnist': (2, 28, 28),
    'dsprites': (64, 64)
}