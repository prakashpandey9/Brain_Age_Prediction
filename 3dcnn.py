# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd
import csv as csv
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Merge,Lambda, Embedding, Bidirectional, LSTM, Dense, RepeatVector, Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy

print (keras.backend.image_data_format())
if True:

    files_all=sorted(glob.glob("/braindata/Combodata/*.nii"))
    
    df = pd.read_csv('/newdata/final.csv')
    y = df.values # A numpy array containing age label of 559 persons
    #print ('\n here \n',y.shape)
    #print ('\n here \n',len(files_all))
    #y=y[:,1]
    y = y.astype(int)
    files_all, test_files, y, test_labels = train_test_split(files_all,y,test_size=0.2)



'''for f in files:
    img = nib.load(f)
    img_data = img.get_data()
    img_data = img_data[:, :, 0:144]
    img_data = np.asarray(img_data)
    trainX.append(img_data)'''
'''x_train = trainX[:-20]
x_train = np.asarray(x_train)

x_test = trainX[-20:]
x_test = np.asarray(x_test)

# The data, shuffled and split between train and test sets:
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')'''

'''df = pd.read_csv('Nfiles/final.csv', header=0)
y = df.values # A numpy array containing age label of 559 persons
y=y[:,1]
y = y.astype(int)
#y = y[:len(trainX)]
#y_train = y[:-20]
#y_test = y[-20:]'''
batch_size = 10
num_classes = 100
epochs = 100
file_size = 110

dimx,dimy,channels = 256,256,144

# Convert class vectors to binary class matrices.

inpx = Input(shape=(dimx,dimy,channels,1),name='inpx')
x = Convolution3D(2, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv1')(inpx)
#x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                       #border_mode='valid', name='pool1')(x)
x = Convolution3D(4, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv2')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                       border_mode='valid', name='pool2')(x)
x = Convolution3D(8, 3, 3, 3, activation='relu',
                        border_mode='same', name='conv3')(x)
x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                       border_mode='valid', name='pool3')(x)

'''model = Sequential()
# 1st layer group
model.add(Convolution3D(2, 3, 3, 3, activation='relu', 
                        border_mode='same', name='conv1',
                        #subsample=(1, 1, 1), 
                        input_shape=(256, 256, 144, 1)))
    
model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                       border_mode='valid', name='pool1'))

model.add(Flatten())'''

hx = Flatten()(x)

#model.add(Dense(4096, activation='relu', name='fc6'))
#model.add(Dropout(.5))
#model.add(Dense(4096, activation='relu', name='fc7'))
#model.add(Dropout(.5))

score = Dense(100, activation='softmax', name='fc8')(hx)

model = Model(inputs=inpx, outputs=score)

opt = keras.optimizers.rmsprop(lr=0.01, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#x_train = np.expand_dims(x_train,axis=0)
#x_train = np.reshape(x_train,(8,256,256,144,1))

model.summary()
#file_size = 10
#for i in range(len(files_all)/file_size):
for i in range(1):
    files = files_all[i*file_size:(i+1)*file_size]
    train_x = []
    for f in files:
        img = nib.load(f)
        img_data = img.get_data()
        img_data = np.asarray(img_data)
        if(img_data.shape==(176,256,256)):
            img_data = img_data.reshape([256,256,176])
        img_data = img_data[:,:,0:144]
        train_x.append(img_data)
    #x_train = train_x[:-2]
    x_train = np.asarray(train_x)
    print('\n iteration number :', i,'\n')
    #x_train = x_train.astype('float32')
    #x_train /= 255
    x_train = np.expand_dims(x_train,4)
    print ('\n', x_train.shape)
    y_train = y[i*file_size:(i+1)*file_size]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    #print ('\n\n shape of img_data \n\n',img_data.shape)
    #print ('\n\n shape of x_train \n\n',x_train.shape)
    #print ('\n\n shape of y_train \n\n',y_train.shape)
    #print ('\n\n')q
    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,verbose=2)
          #validation_data=(x_test, y_test),
          #shuffle=True)

test_x,test_y = [], test_labels

for i,f in enumerate(test_files):
#for i,f in enumerate(test_files[0:10]):
    img=nib.load(f)
    img_data=img.get_data()
    img_data= np.asarray(img_data)
    if(img_data.shape==(176,256,256)):
        img_data=img_data.reshape([256,256,176])
    img_data=img_data[:,:,0:144]
    test_x.append(img_data)

test_x = np.asarray(test_x)
test_x = np.expand_dims(test_x,4)
test_y = keras.utils.to_categorical(test_y, num_classes)
pred = model.predict([test_x])
print('/n', pred.shape)

#pred = model.predict([x_train])
pred = [i.argmax() for i in pred]
#print (pred)

print ('\n assertion::\n',len(pred),len(test_y))

#test_y = [i[0]for i in test_y[0:len(pred)]]
#print (test_y[0:10])


mae = mean_absolute_error(test_y,pred)
pearson = scipy.stats.pearsonr(test_y,pred)
r2 = r2_score(test_y,pred)
mse = mean_squared_error(test_y,pred)

scores =[mae,pearson,r2,mse]

pd.to_pickle(scores,'/output/scores_out')
pd.to_pickle(pred,'/output/pred_out')

print('\n\n MAE is :- ', mae)
print('\n\n pearsonr is:-',pearson)
print('\n\n R2 is :- ', r2)
print('\n\n MSE is :- ', mse)
print('\n\n')
