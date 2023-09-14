import data.cmnist
import data.dsprites
from utils.plot import plot_grayscale_image, plot_red_green_image


N_CLASSES = 2
N_ENVS = 2


MAKE_DATA = {
    'cmnist': data.cmnist.make_data,
    'dsprites': data.dsprites.make_data
}


PLOT = {
    'cmnist': plot_red_green_image,
    'dsprites': plot_grayscale_image
}


IMAGE_SHAPE = {
    'cmnist': (2, 28, 28),
    'dsprites': (data.dsprites.IMAGE_SIZE, data.dsprites.IMAGE_SIZE)
}

X_SIZE = {
    'cmnist': data.cmnist.X_SIZE,
    'dsprites': data.dsprites.X_SIZE
}