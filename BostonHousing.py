from keras.datasets import boston_housing
from keras import models,layers
import numpy as np
from math import *
(train_data,train_prices),(test_data,test_prices) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(train_prices.shape)
print(test_prices.shape)
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std
test_data-=mean
test_data/=std

def housing_model():
    #Using a function due to multiple instantiation
    model=models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#K-fold validation
k=4
num_val_samples=len(train_data)//k
num_epochs=100
all_scores=[]
for i in range(k):
    print('Processing Fold #',i)
    #Prepare the validation data through partitions
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_prices = train_prices[i*num_val_samples:(i+1)*num_val_samples]

    #Training data
    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    partial_train_prices = np.concatenate(
        [train_prices[:i*num_val_samples],
         train_prices[(i+1)*num_val_samples:]],
        axis=0)

    #Build the Keras model
    model = housing_model()
    #train the model
    model.fit(partial_train_data,partial_train_prices,
              epochs=num_epochs,batch_size=1,verbose=0)
    val_mse,val_mae=  model.evaluate(val_data,val_prices,verbose=0)
    all_scores.append(val_mae)

print(all_scores)


    

