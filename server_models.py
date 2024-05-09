import keras
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D, Layer, Add, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, EarlyStopping

import tensorflow as tf
import numpy as np



def digit_nnlock(input_shape=(28,28,1)):
	model=Sequential()
	model.add(Input(shape=input_shape))
	model.add(Conv2D(32, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(512,activation="relu"))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation="softmax"))
	return model
# epochs = 150
# pat = patience(epochs)
# lr_sch = ExponentialDecay(initial_learning_rate=0.005, decay_rate=0.1/epochs, decay_steps=100000)
# opt = SGD(learning_rate=lr_sch, nesterov=False)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
# l = 	EarlyStopping(monitor='val_loss', patience=pat, verbose=1, mode='auto', restore_best_weights=True)
# model , hist = model_utils.train(model, train_ds, val_ds, epochs, [l])


def fashion_nnlock(input_shape=(28,28,1)):
	model=Sequential()
	model.add(Input(shape=input_shape))
	model.add(Conv2D(64, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(256,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation="softmax"))
	return model
# epochs = 180
# pat = patience(epochs)
# lr_sch = ExponentialDecay(initial_learning_rate=0.015, decay_rate=0.1/epochs, decay_steps=100000)
# opt = SGD(learning_rate=lr_sch, nesterov=False)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])	
# l = 	EarlyStopping(monitor='val_loss', patience=epat, verbose=1, mode='auto', restore_best_weights=True)
# model , hist = model_utils.train(model, train_ds, val_ds, epochs, [l])



def cifar10_nnlock(input_shape=(32,32,3)):
	model=Sequential()
	model.add(Input(shape=input_shape))
	model.add(Conv2D(64, kernel_size=3, activation='relu'))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.35))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(Conv2D(256, kernel_size=3, activation='relu'))
	model.add(Conv2D(128, kernel_size=3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.35))
	model.add(Flatten())
	model.add(Dense(512,activation="relu"))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation="relu"))
	model.add(Dropout(0.3))
	model.add(Dense(10,activation="softmax"))
	return model
	# 72.8%
# epochs = 100
# lr_sch = ExponentialDecay(initial_learning_rate=0.0002, decay_rate=0.05/epochs, decay_steps=100000)
# opt = Adam(learning_rate=lr_sch)
# model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# l = 	EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto', restore_best_weights=True)
# model , hist = model_utils.train(model, train_ds, val_ds, epochs, [l])


def flowers102(input_shape=(96,96,3)):
	model=Sequential()
	model.add(Input(shape=input_shape))
	model.add(Conv2D(32, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Conv2D(64, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(128, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(256, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(Conv2D(256, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(Conv2D(512, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1024,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(256,activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(10,activation="softmax"))
	return model
