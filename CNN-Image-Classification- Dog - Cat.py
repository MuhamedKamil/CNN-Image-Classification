import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle
import tensorflow as tf

Data_directory = "D:/Work/Computer Science/Machine learning/Datasets/KaggleCats-Dogs" 
categories = ["Dog","Cat"] 
IMG_SIZE = 64

DataSet = []
X_train = []
Y_train = []
X_test = []
Y_test = []
#================================================================================================
 # read all dataset from 2 files "dog , cat" , reshape [64,64] , covert to grayscale image 
def ReadDataSet ():
    for categorie in categories:
        path = os.path.join (Data_directory,categorie)
        classes = categories.index(categorie)
        for img in os.listdir(path):
            try:
                image_dataset = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                image_dataset = cv2.resize(image_dataset , (IMG_SIZE,IMG_SIZE))
                DataSet.append([image_dataset , classes])
            except Exception as e :
                pass
    
ReadDataSet()
#================================================================================================
#shuffle data
def ShuffleDatasset ():
    random.shuffle (DataSet)
#================================================================================================
# dataset consists of 24,5000 image 
# dividing dataset into train and test set (train --> 22,000 images , test --> 2500)

def GetTrainingData ():
    for features , labels in (DataSet[0:22000]) :
        X_train.append(features)
        Y_train.append(labels)

def GetTestingData ():
    for features , labels in (DataSet[22000:]) :
        X_test.append(features)
        Y_test.append(labels)
#================================================================================================ 
ShuffleDatasset()
GetTrainingData()
GetTestingData()
#================================================================================================
#convert all lists to numpy array and divided by 255 to normlize 
X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1) /255
Y_train = np.array(Y_train)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1) /255
Y_test = np.array(Y_test)
#================================================================================================
#Build model 
model = tf.keras.models.Sequential()
#------------------------------------------------------------------------------------------------
#convolution Layer 1
model.add(tf.keras.layers.Conv2D(256 , [3,3] , input_shape = (IMG_SIZE , IMG_SIZE , 1) , activation= "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
#------------------------------------------------------------------------------------------------
#convolution Layer 2
model.add(tf.keras.layers.Conv2D(256 , [3,3] , activation= "relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2)))
#------------------------------------------------------------------------------------------------
#Fully connected Layer
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64))
model.add(tf.keras.layers.Dense(1 , activation ="sigmoid"))
#------------------------------------------------------------------------------------------------
model.compile(optimizer = "adam" , loss = "binary_crossentropy" , metrics = ["accuracy"])
#------------------------------------------------------------------------------------------------
#Training step
model.fit (x = X_train ,y = Y_train , batch_size= 32 , epochs = 15 )
#------------------------------------------------------------------------------------------------
#Test Accuracy
loss , acc = model.evaluate(X_test , Y_test)
print ("The loss  = ",loss ,"The accuracy = " ,acc)
#------------------------------------------------------------------------------------------------
#Save Model
check_save = input ("Do You want Save model [y/n]")
if (check_save == 'y'):
    model.save("CNN_trainedmodel_Dog_Cat.model")
else:
    pass 
 #------------------------------------------------------------------------------------------------



