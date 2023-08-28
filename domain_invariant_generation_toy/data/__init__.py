import data.colored_mnist
import data.dsprites
from utils.plot import plot_grayscale_image, plot_red_green_image

N_CLASSES = 2
N_ENVS = 2


MAKE_DATA = {
    'colored_mnist': data.colored_mnist.make_data,
    'dsprites': data.dsprites.make_data
}


PLOT = {
    'colored_mnist': plot_red_green_image,
    'dsprites': plot_grayscale_image
}


IMAGE_SHAPE = {
    'colored_mnist': (2, 28, 28),
    'dsprites': (data.dsprites.IMAGE_SIZE, data.dsprites.IMAGE_SIZE)
}