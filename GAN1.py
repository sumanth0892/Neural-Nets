from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models,layers
(trI,Y),(tsI,y) = mnist.load_data()
from keras.utils.np_utils import to_categorical


def build_gen(inputs,image_size):
    #Stack of BN-RELU-CONv2d layers to generate fake images
    #output is sigmoid instead of tanh
    #Sigmoid converges easily
    image_resize = image_size//4
    kernel_size = 5
    layer_filters = [128,64,32,1]
    x = layers.Dense(image_resize*image_resize*layer_filters[0])(inputs)
    x = layers.Reshape((image_resize,image_resize,layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters=filters,kernel_size=kernel_size,
                                   strides=strides,paddimg='same')(x)
    x = layers.Activation('sigmoid')(x)
    generator = models.Model(inputs,x,name='generator')
    return generator

def build_discriminator(inputs):
    #Building a discriminator model
    #Stack to discriminate real from fake
    kernel_size = 5
    layer_filters = [32,64,128,256]

    x = inputs
    for filter in filter_layers:
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size,
                          strides = strides, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    x = layers.Activation('sigmoid')(x)
    discriminator = models.Model(inputs,x,name='discriminator')
    return discriminator

def plot_images(generator,noise_input,show=False,step=0,model_name="GAN"):
    #Generate fake images and plot them
    #Square grid
    os.makedirs(model_name,exist_ok=True)
    filename = os.path.join(model_name,"%0.05d.png"%step)
    images = generator.predict(noise_input)
    plt.figure(figsize=(2.2,2.2))
    num_images = images.shape[0]
    image_size = images.shape[1]
    rows = int(math.sqrt(noise_input.shape[0]))
    for i in range(num_images):
        plt.subplot(rows,rows,i+1)
        image = np.reshape(images[i],[image_size,image_size])
        plt.imshow(image,cmap='gray')
        plt.axis('off')
    plt.save(filename)
    if show:
        plt.show()
    else:
        plt.close('all')

def train(models,x_train,params):
    #Train the two networks
    #Alternatively train discriminator and adversarial networks by batch
    generator,discriminator,adversarial = models
    batch_size,latent_size,train_steps,model_name = params
    save_interval = 500 #Generator image saved every 500 steps
    noise_input = np.random.uniform(-1.0,1.0,size=[16,latent_size])
    train_size = x_train.shape[0]
    for i in range(train_steps):
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_images = x_train[rand_indexes]
        noise = np.random.uniform(-1.0,-1.0,size=[batch_size,latent_size])
        fake_images = generator.predict(noise)
        x = np.concatenate((real_images,fake_images))
        y = np.ones([2*batch_size,1])
        y[batch_size:,:] = 0.0
        loss,acc = discriminator.train_on_batch(x,y)
        log = "%d: [discriminator loss: %f, acc: %f]"%(i,loss,acc)

        #Train the adversarial network for 1 batch
        noise = np.random.uniform(-1.0,1.0,size=[batch_size,latent_size])
        y = np.ones([batch_size,1])
        loss,acc = adversarial.train_on_batch(noise,y)
        log = "%s: [adversarial loss: %f, acc: %f]"%(log,loss,acc)
        print(log)

        if(i+1)%save_interval == 0:
            if (i+1)==train_steps:
                show = True
            else:
                show = False

            plot_images(generator,noise_input=noise_input,
                        show=show,step=(i+1),model_name=model_name)
    generator.save(model_name + ".h5")



def build_and_train_models():
    (x_train,_),(_,_) = mnist.load_data()
    image_size = x_train.shape[1]
    x_train = np.reshape(x_train,[-1,image_size,image_size,1])
    x_train = x_train.astype('float32')/255.0
    model_name = "GAN1_MNIST"
    latent_size = 100
    batch_size=64
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size,image_size,1)

    #Discriminator model
    inputs = layers.Input(shape=input_shape,name='dicriminator_input')
    discriminator = build_discriminator(inputs)
    optimizer = layers.optimizers.RMSprop(lr=lr,decay=decay)
    discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,
                          metrics=['acc'])
    discriminator.summary()

    #Generator model
    input_shape = (latent_size,)
    inputs = layers.Input(shape=input_shape,name='z_input')
    generator = build_generator(inputs,image_size)
    generator.summary()

    discriminator.trainable = False
    adversarial = Model(inputs,discriminator(generator(inputs)),
                        name = model_name)
    adversarial.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=lr*0.5,
                                                                     decay=decay),
                        metrics=['accuracy'])
    adversarial.summary()

    models = (generator,discriminator,adversarial)
    params = (batch_size,latent_size,train_steps,model_name)
    train(models,x_train,params)

def test_generator(generator):
    noise_input = np.random.uniform(-1.0,1.0,size=[16,100])
    plot_images(generator,noise_input=noise_input,show=True,
                model_name = "test_outputs")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g","--generator", help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        test_generator(generator)
    else:
        build_and_train_models()
                   
        
    
