#Load Data
import os
import csv
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
image_directory = "thesis/Color Snippits"
master_file_path = "master_CSV_snips.csv"
X = []
y = []
i = 0
with open(master_file_path) as master_file:
    master_file_stream = csv.reader(master_file, delimiter=",")
    for row in master_file_stream:
        if os.path.exists(image_directory + "/" + row[0]) and row[1] != 'b':
            image = io.imread(image_directory + "/" + row[0])
            image = image / 255 #normalize pixel values (number between 0 and 1).
            X.append(image)
            if row[1] == 's':
                new_y = [1,0,0,0,0]
            elif row[1] == 'j':
                new_y = [0,1,0,0,0]
            elif row[1] == 'm':
                new_y = [0,0,1,0,0]
            elif row[1] == 'p':
                new_y = [0,0,0,1,0]
            else:
                new_y = [0,0,0,0,1]
            y.append(new_y)
            i += 1
        
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Partition Data
X_buckets = list()
y_buckets = list()
for bucket_number in range(0,10):
    X_bucket = list()
    y_bucket = list()
    for index in range (0,int(len(y_train)/10)):
        X_bucket.append(X_train[bucket_number * index + index])
        y_bucket.append(y_train[bucket_number * index + index])
    X_buckets.append(X_bucket)
    y_buckets.append(y_bucket)


#Instantiate and train models
from keras.models import Sequential
from keras.layers import Conv2D, AvgPool2D, Activation, Flatten, Dense, MaxPool2D
from keras.optimizers import SGD

X=None
y=None
from keras.applications.resnet50 import ResNet50
from keras.optimizers import SGD
for i in range(0,10):
    validation_X = X_buckets[i]
    validation_y = y_buckets[i]
    training_X = list()
    training_y = list()
    for x in range(0,10):
        if x != i:
            training_X.extend(X_buckets[x])
            training_y.extend(y_buckets[x])
    validation_X = np.array(validation_X)
    validation_y = np.array(validation_y)
    training_X = np.array(training_X)
    training_y = np.array(training_y)
    model = Sequential([
            Conv2D(input_shape=(31, 31, 3), filters=96, kernel_size=(4, 4), strides=(1, 1), padding='valid'),
            Activation('relu'),
            MaxPool2D(pool_size=(3, 3), strides=(2,2)),
            Conv2D(filters=256, kernel_size=(3,3), strides=(1, 1), padding='valid'),
            Activation('relu'),
            MaxPool2D(pool_size=(3, 3), strides=(1,1)),
            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'),
            Activation('relu'),
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'),
            Activation('relu'),
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'),
            Activation('relu'),
            MaxPool2D(pool_size=(2,2), strides=(1,1)),
            Flatten(),
            Dense(100),
            Activation('relu'),
            Dense(100),
            Activation('relu'),
            Dense(5),
            Activation('softmax')
        ])
    optimizer = SGD(lr=10e-3)
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    print("go")
    model.fit(x=training_X,y=training_y,epochs=120,validation_data=(validation_X, validation_y),batch_size=1,verbose=1)
    model.save("AlexNet_Species_Phase1_" + str(i))
    number_right = 0
    for X_val, y_val in zip(X_test, y_test):
        value = model.predict(np.array([X_val]))[0].tolist()
        if np.argmax(value) == np.argmax(y_val):
            number_right += 1
    print("~~~~~~~~~~~test accuracy: " + str(float(number_right)/float(len(y_test))))
