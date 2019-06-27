# To ignore annoying warning
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
warnings.filterwarnings("ignore", category=DeprecationWarning)
print('Deprecation warning will be ignored!')

import os
from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.callbacks import EarlyStopping,LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras.layers import Dense,Dropout,BatchNormalization, Activation
from keras import models
from keras.models import Sequential
from sklearn.metrics import accuracy_score
import random

from keras.optimizers import Adam
from keras.utils import np_utils
import keras as ks

from pylab import rcParams
rcParams['figure.figsize'] = 8, 8

FILE_NAME = 'Kaggle_Pokamon'
save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
def build_model(shape):
    model = ks.Sequential()
    n = 512
        
    for i in range(6):
        if i == 0:
            model.add(Dense(n, input_dim=shape))
        else:
            model.add(Dense(n))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(6))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Activation('softmax'))
#     print(model.summary())
    return model


def display_curve(model_history):
    train_loss = model_history.history['loss']
    train_acc = model_history.history['acc']
    valid_loss = model_history.history['val_loss']
    valid_acc = model_history.history['val_acc']
    plt.subplots(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'b', label='train')
    plt.plot(valid_loss, 'r', label='valid')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'b', label='train')
    plt.plot(valid_acc, 'r', label='valid')
    plt.legend(loc=4)
    plt.title("Accuracy")
    plt.show()

def auto_train(X, y, 
               test_x,
               test_y,
               model_name,
               batch_size=256, 
               epochs=200,  
               max_loops=20):
    
    count = 0
    best_model = []
    filepath = []
    while count < max_loops:
        print ("\n\nLoop %d:\n"%(count))
        train_x, val_x, train_y, val_y = train_test_split(X, 
                                                          y, 
                                                          test_size=0.1,
                                                          random_state=int(time()))
        model = build_model(X.shape[1])

        model.compile(Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        early_stopping = ks.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=50,
                                                    mode='min',
                                                    verbose=1) 

        filepath = FILE_NAME + model_name + str(count) + ".hdf5"
        
        # 檢查點設定，監控 val_loss 
        checkpoint = ks.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, filepath), 
                                                  monitor='val_loss', 
                                                  save_weights_only=True,
                                                  verbose=0,
                                                  mode='min',
                                                  save_best_only=True)
        # 動態調整 learning rate
        lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       patience=3,
                                       min_lr=0.5e-6)

        model_history = model.fit(x=train_x, y=train_y,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  shuffle=True,
                                  verbose=0,
                                  validation_data=(val_x,val_y),
                                  callbacks=[checkpoint, 
                                             lr_reducer, 
                                             early_stopping
                                            ]
                                 )
        # 繪製學習曲線
        display_curve(model_history)
        
        
        # 重載當下最佳 Model (val_loss最低)
        model.load_weights(os.path.join(save_dir,filepath))
        
        # 評估成效 in valid datasets 
        val_loss, val_accuracy = model.evaluate(val_x, val_y) 
        print('[Valid] loss:%4f, accuracy:%4f'%(val_loss, val_accuracy))
        
        # 評估成效 in testing datasets 
        test_loss, test_accuracy = model.evaluate(test_x, test_y) 
        print('[Test] loss:%4f, accuracy:%4f'%(test_loss, test_accuracy))
    
        # 儲存最佳 Model (Test acc 超過 0.55)
        if test_accuracy > 0.55:
            print("Save best model(test_acc:%4f) ..."%(test_accuracy))
            best_model.append(model)
        count += 1
            
    return best_model