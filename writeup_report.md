
# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn_architecture.png "CNN Architecture"
[image2]: ./examples/center_lane_driving.png "Center lane driving"
[image3]: ./examples/left_camera.png "Left Camera"
[image4]: ./examples/center_camera.png "Center Camera"
[image5]: ./examples/right_camera.png "Right Camera"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
My model architecture is based on the behavioral cloning CNN found here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/.

My model consists of a convolution neural network with three 5x5 convolutions, two 3x3 convolutions, one flatten layer, and 3 fully connected layers.
The following image illustrates the model architecture.  

&nbsp;&nbsp;&nbsp;&nbsp;
![alt text][image1]
&nbsp;&nbsp;&nbsp;&nbsp;  
Note:
* Every convolution layer includes RELU activation to introduce nonlinearity.
* Normalization is accomplished using a Keras lambda layer. 
* The images cropped using a Cropping 2D layer. This is done to eliminete irrelevant backround data such as trees and sky pixels.



#### 2. Attempts to reduce overfitting in the model

* The model was trained and validated on different data sets
* Only 1 epoch was used
* The model was tested in the simulator, which ensured that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model was trained using an adam optimizer, so the learning rate was not tuned manually. The adam optimizer was able to produce minimal validation error on the first epoch. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and  smooth turning. For both of these cases images from all three cameras were used (with an angle correction factor applied in preprocessing). Using the lateral camera images enable the model to learn to recover after veering off the center. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for solving the problem was to start with an architecture that has proven effective for a similar task, and then modifying and tuning the model to increase performance. 

My first step was to use a convolution neural network model similar to the NVIDIA behavioral cloning CNN found here: https://devblogs.nvidia.com/deep-learning-self-driving-cars/. This model seemed like a great starting point because the goal of NVIDIA model is similar to the goal of this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that the model had a low mean squared error on both the traning and validation set after minimal training. This was expected because this model architecture has proven effective for similar tasks.

The model satsfied the project requirements so no major modifications to the architecture were made.

The model's performance was tested in the simulator, which showed how the car would operate. Initially, there were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I augmented the training data by adding: 
* Mirror Images with a mirrored (negative) steering value
* Left camera images with a correction factor (+0.2)
* Right camera images with a correction factor (-0.2)

After training the model on the augmented data set, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture was based on the Nvidia architecture presented earlier in this writeup. 
The model consists of a convolution neural network with three 5x5 convolutions, two 3x3 convolutions, one flatten layer, and three fully-connected layers.
The following table, produced by the Model.summary() function, illustrates the model architecture. 


Layer (type)        			Output Shape)       Param #        
_________________________________________________________________
input_1 (InputLayer)         (None, 160, 320, 3)       0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11      
_________________________________________________________________

Total params: 981,819
Trainable params: 981,819
Non-trainable params: 0


#### 3. Creation of the Training Set & Training Process

The sample training data provided proved sufficient to accomplish the task. 

To capture good driving behavior, the data included mostly center-lane driving. Smooth turning data was also recorded. Here is an example image of center lane driving:

![alt text][image2]

The data also included left and right camera images with corrected steering values. Using the images from the left and right cameras enabled the vehicle to learn to recover smoothly after falling off the center line. These images show the left, center, and right camera images.

![alt text][image3]  

![alt text][image4]  

![alt text][image5]


To augment the data set, center camera images were flipped and the steering values were muliplied by -1. This augmentation counteracts the data bias towards left turns that comes from driving clockwise around the track. Here is an example of a flipped image:

![alt text][image6]
![alt text][image7]

Training Summary:
1. The sample training data had 8036 data points. 
2. After augmentation, the amount of training data increased to 32144. 
3. I preprocessed this data by normalizing, mean centering, and cropping the pixel data.
4. I randomly shuffled the data set and put 20% of the data into a validation set.
5. I implemented generators to efficiently load and preprocess the data.
6. I trained the model using the preprocessed training data. The validation set helped determine if the model was over or under fitting. 
7. The ideal number of epochs was 1 as evidenced by the validation loss being the lowest afer the first epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary. 
