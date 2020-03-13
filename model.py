from keras.layers import Lambda, Input, Cropping2D, Dense, Conv2D, Flatten
from keras.models import Model
import tensorflow as tf
import csv 
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
from math import ceil
from sklearn.utils import shuffle
import cv2
# Read csv data
lines = []
with open('/opt/carnd_p3/data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

# labels: ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
# example entry: ['IMG/center_2016_12_01_13_30_48_287.jpg', ' IMG/left_2016_12_01_13_30_48_287.jpg', ' IMG/right_2016_12_01_13_30_48_287.jpg', ' 0', ' 0', ' 0', ' 22.14829']

# #  Splitting Data
data_lines = lines[1:]
training_samples, validation_samples = train_test_split(data_lines, test_size=0.2, shuffle = True)


def validation_generator (samples, batch_size = 100):
    # used to generate validation data
    num_samples = len(samples) 
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:batch_size+offset]
            images = []
            measurements = []

            for line in batch_samples:
                center_filename = line[0].split('/')[-1]
                center_image_path =  './data/IMG/' + center_filename
                center_image = ndimage.imread(center_image_path)

                steering_center = float(line[3])

                images.append(center_image)
                measurements.append(steering_center)
            X_val = np.array(images)
            y_val = np.array(measurements)
            yield shuffle(X_val, y_val)

def augmented_training_generator (samples, batch_size = 100):
    # used to generate training data. Incorporates augmentation
    num_samples = len(samples) 
    while 1:
        shuffle(samples)
        correction_factor = 0.2 #correction factor for left and right cameras
        # 4 images are produced from each data line with augmentation
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:batch_size+offset]
            images = []
            measurements = []
            for line in batch_samples:
                center_filename = line[0].split('/')[-1]
                left_filename = line[1].split('/')[-1]
                right_filename = line[2].split('/')[-1]
                center_image_path =  './data/IMG/' + center_filename
                left_image_path = './data/IMG/' + left_filename
                right_image_path = './data/IMG/' + right_filename
                center_image = ndimage.imread(center_image_path)
                left_image = ndimage.imread(left_image_path)
                right_image = ndimage.imread(right_image_path) 
                mirror_image = np.fliplr(center_image)
                steering_center = float(line[3])
                steering_left = steering_center + correction_factor
                steering_right = steering_center - correction_factor
                steering_mirror = -1*steering_center 
                images.extend((center_image,left_image, right_image, mirror_image))
                measurements.extend((steering_center, steering_left, steering_right, steering_mirror))

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
    
# Defining Keras model architecture

# Preprocessing Layers
image_height = 160
image_width = 320 
color_channels = 3
inputs= Input(shape=(image_height,image_width,color_channels))
# Normalizing and mean centering the data 
normalized = Lambda(lambda x: x/255-0.5) (inputs)
# Cropping layer: new image size = 90x320          
cropped_inputs = Cropping2D(cropping = ((50, 20), (0, 0)))(normalized)

# Convolution layers
conv1 = Conv2D(24, 5, strides=(2,2), activation = 'relu')(cropped_inputs)
conv2 = Conv2D(36, 5, strides=(2,2), activation = 'relu')(conv1)
conv3 = Conv2D(48, 5, strides=(2,2), activation = 'relu')(conv2)
conv4 = Conv2D(64, 3, activation = 'relu')(conv3)
conv5 = Conv2D(64, 3, activation = 'relu')(conv4)

#flatten layer
flatten = Flatten()(conv5)

#fully connected layers
fc1 = Dense(100)(flatten)
fc2 = Dense(50)(fc1)
fc3 = Dense(10)(fc2)

#steering angle prediction
prediction = Dense(1)(fc3)

# Compiling Model
model = Model(inputs = inputs, outputs = prediction)
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])                                                 
model.summary()


#Training and Saving Model

batch_size = 100
epochs = 1
t_generator = augmented_training_generator(training_samples, batch_size)
v_generator = validation_generator(validation_samples, batch_size)  

model.fit_generator(t_generator, 
            steps_per_epoch=ceil(len(training_samples)/batch_size), 
            validation_data=v_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=epochs, verbose=1)

# model.save('model.h5')
print('done')

