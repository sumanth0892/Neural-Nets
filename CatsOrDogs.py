import os,shutil
#Downloaded data location
original_dataset_direc = ' '
#Where the smaller data will be stored
base_direc = ' '
os.mkdir(base_direc)

#TRAINING DIRECTORIES
#Validation and test splits
training_direc = os.path.join(base_direc,'train')
os.mkdir(training_direc)
validation_direc = os.path.join(base_direc,'validation')
os.mkdir(validation_direc)
test_direc = os.path.join(base_direc,'test')
os.mkdir(test_direc)

#Directory with training pictures of cats
train_cats_direc = os.path.join(training_direc,'cats')
os.mkdir(train_cats_direc)

#Directory with training pictures of dogs
train_dogs_direc = os.path.join(training_direc,'dogs')
os.mkdir(train_dogs_direc)

#Directory with Validation pictures of cats
validation_cats_direc = os.path.join(validation_direc,'cats')
os.mkdir(validation_cats_direc)

#Directory with validation pictures of dogs
validation_dogs_direc = os.path.join(validation_direc,'dogs')
os.mkdir(validation_dogs_direc)

#Directory with test pictures of cats
test_cat_direc = os.path.join(test_direc,'cats')
os.mkdir(test_cat_direc)

#Directory with test pictures of dogs
test_dog_direc = os.path.join(test_direc,'dogs')
os.mkdir(test_dog_direc)

#Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_direc,fname)
    dst = os.path.join(train_cats_direc,fname)
    shutil.copyfile(src,dst)

#Copy next 500 cat images to validation directory
fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_direc,fname)
    dst = os.path.join(validation_cats_direc,fname)
    shutil.copyfile(src,dst)

#Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_direc,fname)
    dst = os.path.join(train_dogs_direc,fname)
    shutil.copyfile(src,dst)

#Copy next 500 cat images to validation directory
fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_direc,fname)
    dst = os.path.join(validation_dogs_direc,fname)
    shutil.copyfile(src,dst)
    
#Similarly other files are transferred to respective directories



from keras import layers,models
CatOrDog = models.Sequential()
CatOrDog.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(64,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(128,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(128,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Flatten())
CatOrDog.add(layers.Dense(512,activation='relu'))
CatOrDog.add(layers.Dense(1,activation='sigmoid'))
from keras import optimizers
CatOrDog.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=1e-4),metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
training_data = ImageDataGenerator(rescale=1./255)
testing_data = ImageDataGenerator(rescale=1./255)
training_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validating_generator = test_datagen.flow_from_directory(validating_dir,target_size=(150,150),batch_size=20,class_mode='binary')
history = model.fit_generator(training_generator,steps_per_epoch=100,epochs=30,validation_data=validating_generator,validation_steps=50)
model.save('CatsOrDogsmicro.h5')

#Plotting the display curves of loss and accuracy
import matplotlib.pyplot as plt
accuracy=history.history['acc']
validation_acc=history.history['val_acc']
loss=history_history['loss']
validation_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'ro',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
