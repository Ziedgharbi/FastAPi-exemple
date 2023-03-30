import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2 as cv

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from importlib import reload

import random
from itertools import repeat

from numba import jit, cuda
import time


project_path="C:/Users/pc/Nextcloud/Python/GITHUB/FastApi-app"
model_path=project_path+'/models/'

def data_load():
    (x_train,y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
    
    return(x_train, y_train, x_test, y_test)


@jit(target_backend='cuda') 
def main():
    
    x_train, y_train, x_test, y_test=data_load()

    x_train=x_train.reshape(-1,28,28,1)  # rajouter 1 parceque CNN besoin de profondeur qui est le niveau de couleur de l'image
    x_test=x_test.reshape(-1,28,28,1)  # de la même façon

    print("X train set dimensions --- ", x_train.shape)
    print("X test set dimensions --- ", x_test.shape)

    print("y train set dimensions --- ", y_train.shape)
    print("y test set dimensions --- ", y_test.shape)


    ## normalisation 
    xmax=x_train.max()

    x_train=x_train/xmax
    x_test=x_test/xmax

    # quelque exemple de la data
    num_row = 2
    num_col = 5
    num =random.sample(range(0,x_train.shape[0]), 10)

    fig = plt.figure(figsize=(1.5*num_col,2*num_row))

    j=1
    for i in num:
        
        image=x_train[i]
        label=y_train[i]
        plt.subplot(num_row,num_col,j)
        plt.imshow(image,cmap='gray_r')
        plt.title(str(label))
        j=j+1
        
    plt.show()

    # model architecture 
    model=keras.models.Sequential()

    model.add(keras.layers.Input((x_train.shape[1],x_train.shape[2],x_train.shape[3])))

    model.add( keras.layers.Conv2D (8,(3,3), activation='relu' ) ) # 8 Convential layers with kernal size 3x3
    model.add(keras.layers.MaxPooling2D((2,2)))  #  reduction of image dimension
    model.add(keras.layers.Dropout(0.2))        # desactivate some neural (20% output neural puted on 0)
                                                # force learning on all neural, so more effiency

    model.add ( keras.layers.Conv2D(16, (3,3), activation='relu') )
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation="softmax"))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(len(np.unique(y_train)), activation="softmax")) # output results : number of differents classes of target

    model.summary()


    # model compile 
    model.compile(optimizer='adam',
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    #training
    batch_size = 512
    epochs     = 16

    history = model.fit (x_train, y_train, 
                         batch_size= batch_size,
                         epochs = epochs ,
                         verbose = 1,
                         validation_data=(x_test, y_test))
    
    model.save(model_path)

    score=model.evaluate(x_test, y_test, verbose=0)
    
    fig = plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.plot(range(1,17), history.history["loss"], label='train')
    plt.plot(range(1,17), history.history["val_loss"], label='test')
    plt.title( 'Loss score ')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(range(1,17), history.history["accuracy"], label='train')
    plt.plot(range(1,17), history.history["val_accuracy"], label='test')
    plt.title("Accuracy score")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
      
    print('Test loss :',  score[0])
    print('Test accuracy :' , score[1])
    
    # confusion matrix 
    y_sigmoid=model.predict(x_test)
    y_pred=np.argmax(y_sigmoid, axis=-1)
    mat=confusion_matrix(y_test, y_pred)
    
    mat_plot=ConfusionMatrixDisplay(mat)
    mat_plot.plot()
    plt.show()
      
    # plot some errors 
    error=[i for i in range(len(x_test)) if y_test[i]!=y_pred[i]]
    rand=random.sample(error, 10)
    
    num_row = 2
    num_col = 5
   

    fig = plt.figure(figsize=(1.5*num_col,2*num_row))
    j=1
    for rand in rand:
        
        image=x_train[rand]
        label=str(y_train[rand])+" ( "+str(y_pred[rand])+" )"
        
        plt.subplot(num_row,num_col,j)
        plt.imshow(image,cmap='gray_r')
        plt.xlabel(label)
        
        j=j+1
        
    fig.suptitle('Reall value vs predicted (between brackets)')    
    plt.show()
    
    

    return(history)
    


main()

