""" This module contains various machine learning based functions. """

from tensorflow import keras
import numpy as np

from sbpy import utils
from sbpy import grid2d

N = 21
blocks = utils.get_annulus_grid(N)
coarse_grid = grid2d.MultiblockSBP(blocks, accuracy=4)
coarse_nodes = utils.boundary_layer_selection(coarse_grid, [1,3,5,7], 4)

N = 161
blocks = utils.get_annulus_grid(N)
fine_grid = grid2d.MultiblockSBP(blocks, accuracy=4)
utils.fetch_highres_data(coarse_grid, coarse_nodes, fine_grid, fine_data, 8)



Nc = 21
Nf = 161

model = keras.models.Sequential()

model.add(keras.layers.Dense(16, input_shape=(1,)))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(16))
model.add(keras.layers.Dense(Nc*4*4, activation='linear'))

print(model(np.array([[1],[2]])))
