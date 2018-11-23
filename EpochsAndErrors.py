#Calculate the losses for varying number of epochs on the IMDB movie dataSet
from keras.datasets import imdb
import numpy as np
(traning_data,training_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((sequences,dimension))
    #Convert into a vector of 1s and 0s
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return
x_train = vectorize_sequences(training_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray('training_labels')
y_test = np.asarray('test_labels')

from keras import models,layers
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
partial_x_train = x_train[10000:]
partial_y_train = y_train[10000:]
x_val = x_train[:10000]
y_val = y_train[:10000]

E = [20,22,24,26,30]
for i in range(len(E)):
    history = model.fit(partial_x_train,partial_y_train,epochs=E[i],batch_size=512,validation_data=(x_val,y_val))
    predicts = model.predict(x_test)
    errors = y_test - x_test
    Errs.append(sum(errors))

#Data Can be plotted
import matplotlib.pyplot as plt
.......
.......
.......
.......




    
