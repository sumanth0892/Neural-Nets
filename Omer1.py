import os
import numpy as np
from keras import models,layers
model = models.Sequential()
model.add(layers.Conv2D(96,(11,11),strides=(4,4),activation='relu',input_shape=(224,224,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(256,(5,5),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(384,(3,3),activation='relu'))
model.add(layers.Conv2D(384,(3,3),activation='relu'))
model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dense(4096,activation='relu'))
model.add(layers.Dense(1000,activation='softmax'))
model.summary()
