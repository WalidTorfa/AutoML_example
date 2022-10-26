import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
import keras_tuner as kt

#importing the dataset
mnist= tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test= x_train/255, x_test/255

#using AutoML to optimize the dense layer in a basic model
def model_builder(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    hp_units= hp.Int('units',min_value=16,max_value=512,step=16)
    model.add(Dense(units=hp_units,activation="relu")) #we will optimize the number of units in the dense layer
    model.add(Dropout(0.2))
    model.add(Dense(10,activation="softmax"))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
#We will make optimizing the Val accuracy our target in this example using the Hyperband method to do so
tuner=kt.Hyperband(model_builder,objective='val_accuracy',max_epochs=10,factor=3,directory='Example_dir',project_name='Example')

#Run the AutoML
tuner.search(x_train,y_train,epochs=50,validation_split=0.2)

#incase you want to add
#configer an early stop

stop_early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
#now this line of code confiered an EarlyStopping once to stop the search for the once it sees that the val accuracy stopped improving for 5 max_epochs
#you should add it to the tuner.search if you want to use it. pelase check the below code
#tuner.search(x_train,y_train,epochs=50,validation_split=0.2,callbacks=[stop_early])
