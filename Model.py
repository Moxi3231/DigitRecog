import tensorflow as tf
from keras.datasets import mnist

import numpy as np

def load_Data():
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    train_x,test_x = np.expand_dims(train_x,axis=-1)/255,np.expand_dims(test_x,axis=-1)/255
    train_x,test_x = train_x.astype(dtype=np.float32),test_x.astype(dtype=np.float32)

    temp_x = np.zeros((train_y.size,train_y.max()+1))
    temp_x[np.arange(train_y.size),train_y] = 1
    train_y = temp_x
    return train_x,train_y,test_x,test_y

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=8,kernel_size=(3,3),activation=tf.keras.activations.relu),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32,activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(10,activation=tf.keras.activations.softmax)
    ])


    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(1e-2, 1875*5, 1e-6, power=0.95)
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=1.0, epsilon=1e-08),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=[
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Precision()
    ])
    return model

def get_trained_model(training = True):

    model = get_model()
    saved_Dir = "ModelDigit"
    train_x,train_y,test_x,test_y = load_Data()
    
    if training:
        model.fit(train_x,train_y,epochs = 5)
        model.summary()
        model.save(saved_Dir)
    else:
        model = tf.keras.models.load_model(saved_Dir)
    prd = model.predict(test_x,verbose=0)
    p = np.argmax(prd,axis=1)
    print("Model Accuracy of Test Set (In Percentage):",(np.where(p==test_y)[0].shape[0]/test_x.shape[0])*100)

    return model

get_trained_model(False)