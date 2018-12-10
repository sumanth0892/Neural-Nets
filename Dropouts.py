from keras.datasets import imdb
from keras import models,layers,optimizers,losses
(training_data,training_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

x_train = vectorize_sequences(training_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(training_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
AccVal=[]
LossVal=[]
do = [0.0,0.1,0.2,0.3,0.4,0.5]

for i in range(len(do)):
    model=models.Sequential()
    model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
    model.add(layers.Dropout(do[i]))
    model.add(layers.Dense(16,activation='relu'))
    model.add(layers.Dropout(do[i]))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    x_val=x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val=y_train[:10000]
    partial_y_train = y_train[10000:]
    history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,batch_size=512,
                    validation_data=(x_val,y_val))
    acc=history.history['acc']
    val_acc=history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    AccVal.append(val_acc)
    LossVal.append(val_loss)

plt.plot(do,AccVal,'ro')
plt.title('Dropout vs Validation accuracy')
plt.grid(True)
plt.xlabel('Dropout rate')
plt.ylabel('Validation accuracy')
#Plot for validation loss as well
