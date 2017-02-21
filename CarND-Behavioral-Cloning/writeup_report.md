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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

Each convolutional layer is followed with RELU activation to introduce nonlinearity.

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

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put 10% of the data into a test set and 25% into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
