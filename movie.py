#Classifying movie reviews from IMDB
#The dataset contains 50000 reviews which are highly polarized
from keras.datasets import imdb
import numpy as np
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
#Keep the 10000 most occurring words in the dataSet
#Decoding the integer sequences
word_index = imdb.get_word_index()
#Mapping integers to words
reverse_word_index  =dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(1-3,'?') for i in train_data[0]])

#The data is prepared into a matrix of binary values
def vectorized_seq(sequences,dim=10000):
    #Create an all-zero matrix of shape(len(sequences),dimension)
    results = np.zeros(len(sequences,dim))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

#Training data
x_train = vectorized_seq(train_data)
#Test data
x_test = vectorized_seq(test_data)

#vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

from keras import optimizers,losses,metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=['accuracy'])
#Custom losses and metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])



