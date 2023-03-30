import sys
sys.path.append('../..')
import pickle

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import numpy as np

from sbpy import grid2d
from sbpy import multiblock_solvers
from sbpy import utils

N = 21
blocks = utils.get_annulus_grid(N)
coarse_grid = grid2d.MultiblockGridSBP(blocks, accuracy=4)

N = 161
blocks = utils.get_annulus_grid(N)
fine_grid = grid2d.MultiblockGridSBP(blocks)

train_x = []
train_y = []

for k in range(30):
    filename = "highres_data/highres_sol161_" + str(k) + ".pkl"
    with open(filename, 'rb') as f:
        U_highres,diffusion = pickle.load(f)

    nodes    = utils.boundary_layer_selection(coarse_grid, [1,3,5,7], 4)
    int_data = utils.fetch_highres_data(coarse_grid,
                                        nodes,
                                        fine_grid,
                                        U_highres,
                                        stride=8)
    train_x.append(diffusion)
    train_y.append(int_data)

num_copies = 2000
train_x = num_copies*train_x
train_y = num_copies*train_y

train_x = np.array(train_x)
train_y = np.array(train_y)

model    = models.Sequential()

model.add(layers.Dense(64,activation='relu',input_shape=(1,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(4*4*21, activation="linear"))

model.compile('adam','mse')
model.fit(train_x,train_y,epochs=30)
model.save('bd_layer_predictor.h5')
