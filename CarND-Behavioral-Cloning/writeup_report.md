#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./example_images/model.png "Model Visualization"
[image2]: ./example_images/Y_channel.png "Y channel"
[image3]: ./example_images/cropped_image.png "Cropped Image"
[image4]: ./example_images/resized_image.png "Resized Image"
[image5]: ./example_images/center_image.png "Center Image"
[image6]: ./example_images/normal_recovery.png "Normal Recovery"
[image7]: ./example_images/bridge_recovery.png "Bridge Recovery"
[image8]: ./example_images/hazard_recovery.png "Hazard Recovery"
[image9]: ./example_images/left_right_images.png "Left Right Images"
[image10]: ./examples_images/history.png "Training History"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* preprocess.py containig util function for preprocessing and image loading
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of five convolutional layers with 3x3 filter sizes and depths 8, 16, 24, 32, 32 (model.py lines 58-66). 

Each convolutional layer is followed with RELU activation to introduce non-linearity.

The model has five fully connected layer with 100, 50, 20, 10, 1 neurons (model.py lines 72-83). 

To prevent overfitting, I also added a Dropout layer between two fully connected layers (model.py line 75).

The data is normalized in the model using a Keras lambda layer (code line 55). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 75). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 111-112). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

Also I used left and right camera to create more recovery training data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to overfit the model first. Then based on training and validation loss history adjust the layers to get a good model.

My first step was to use a convolution neural network model similar to the NVIDIA End to end learning model. I thought this model might be appropriate because it contains enough convolutional layers to extract necessary features and combine them with multiply fully connected layers. The model is complicated enough for my first goal: overfitting the model.

In order to gauge how well the model was working, I split my image and steering angle data into training, validation and test sets. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the validation loss is close to training loss. So does the test loss. For the model to give a general average loss more data is needed (I was using the example data which only has 8000 pieces of images angle pairs. So I recorded more data to append to the example data.

Then I ran the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (e.g. bridge). I also found that the model cannot steel the car back when it drove to the side of the road. So I recorded some recovery data by driving the car to the side of the road and recording my recovery behavior. I did this along the track and repeated several times at failing spots with different car poses. The result of adding recovery data is pretty obvious, but still not good enough for some part of the track. To avoid speeding more time on data collection, I added left and right camera images paired with corrected angles. This solves the recovery problem. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 47-88) consisted of a convolution neural network with the following layers and layer sizes

Here is a visualization of the architecture:

![alt text][image1]

####3. Creation of the Training Set & Training Process

Preprocessing:

Get the hint from Traffic Sign Recognition Project, I first convert images from RGB to YUV, and only use Y channel to train the model, the performance is better than using three channel and speed is faster:

![alt text][image2]

By checking the images, I find that top part of the image captures trees, mountains and sky, and the bottom part of the image captures the hood of the car. These information will distract and slow down my model. So I cropped each image to focus only on the road part:

![alt text][image3]

After cropping the images, my GPU is still not able to train the model even with generator. So I resize the image by 1/5:

![alt text][image4]

To capture good driving behavior, I first recorded ten laps on track one using center lane driving. For better visualization, I'll show the image before resize:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself when it deviates from the center. I record some recovery along normal track:

![alt text][image6]

And Special cases:

![alt text][image7]
![alt text][image8]

Besides record recovery images, I also used images from left and right cameras. For left images, I added 0.1 correction to the angle, and for the right images, I subtract 0.1 correction:

![alt text][image9]

To augment the data set, I also flipped images vertically. Because the curve part in track one is mostly curve to the right. Adding flipped images may improve the performance on the last left turn. But I got no luck. So instead, I record more data on the left turn. And the robot is able to pass the left turn smoothly.

After the collection process, I had 27826 number of data points.

I finally randomly shuffled the data set and put 10% of the data into a test set and 25% into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the training and validation loss tend to stabilize. Here's the training and validation loss history:

![alt text][image10]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

Here's the link to my successful run video:
https://youtu.be/N-EutQSFQOI
