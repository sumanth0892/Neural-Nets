from keras.datasets import imdb
from numpy import *
(train_words,Y_train),(test_words,Y_test) = imdb.load_data(num_words=10000)
def vec_seq(sequences,dimension=10000):
    results = zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results
X_train = vec_seq(train_words); X_test = vec_seq(test_words)

from keras import regularizers,models,layers
mod1 = models.Sequential()
mod1.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
mod1.add(layers.Dense(64,activation='relu'))
mod1.add(layers.Dense(1,activation='sigmoid'))
mod1.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='rmsprop')
mod2 = models.Sequential()
mod2.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))
mod2.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
mod2.add(layers.Dense(1,activation='sigmoid'))
mod2.compile(loss='binary_crossentropy',metrics=['acc'],optimizer='rmsprop')
H1 = mod1.fit(X_train,Y_train,epochs=20,batch_size=128,
              validation_split=0.2,verbose=1)
H1_history = H1.history
test1_loss = H1_history['loss']
test1_acc = H1_history['acc']
H2 = mod2.fit(X_train,Y_train,epochs=20,batch_size=128,
              validation_split=0.2,verbose=1)
H2_history = H2.history
test2_loss = H2_history['loss']
test2_acc = H2_history['acc']
epochs = range(1,len(test1_loss)+1)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(epochs,test1_loss,'bo',label='Loss without regularization')
plt.plot(epochs,test2_loss,'ro',label='Loss with regularization')
plt.xlabel('Epochs')
plt.ylabel('Losses')
plt.grid(True)
plt.title("Losses with and without regularization")
plt.legend()
plt.figure()
plt.plot(epochs,test1_acc,'b',label='Accuracy without regularization')
plt.plot(epochs,test2_acc,'r',label='Accuracy with regularization')
plt.xlabel('Epochs')
plt.ylabel('Accuracies')
plt.grid(True)
plt.title("Accuracies of model with and without regularization")
plt.legend()
plt.show()
