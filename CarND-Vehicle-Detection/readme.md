[//]: # (Image References)
[image1]: ./images/car_notcar.png
[image2]: ./images/Car-channel1.png
[image3]: ./images/not-Car-channel1.png
[image4]: ./images/sliding_window.png
[image5]: ./images/example1.png
[image6]: ./images/example2.png
[image7]: ./images/box1.png
[image8]: ./images/box2.png
[image9]: ./images/box3.png
[image10]: ./images/box4.png
[image11]: ./images/box5.png
[image12]: ./images/detected_gray.png
[image13]: ./images/detected.png
[video1]: ./project_video_result.mp4

Please run train.py first to train and save the model.
And then run main.py to generate result video.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in `feature_extraction.py`

I started by reading in all the `vehicle` and `non-vehicle` images. Although the number of `vehicle` and `non-vehicle` images are different by 176. But the difference is relatively small compare to total number of image. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and decided to use orient = 9, pix_per_cell = 8, cell_per_block = 2 and hog_channel = 'ALL'. For orientation, I tried higher values but the model overfits. pix_per_cell and cell_per_block are set to relatively low values to provide enough precision due to the height of ROI is only 256. I used all three channels of YCrCb to give enough features. 

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training of my SVM classifier is defined in function `train_svm_clf` in `train.py`. First I extracted features from both car and noncar images. Hog, spatial histogram and color histogram features were concatenated. Then I used StandardScaler to scale the features into 0 mean and unit variance. `Train_test_split` function was used to seprate training and testing data. Then I used the Linear SVM model as my classifier. Model and feature hyper parameters were saved into `svc_pickle.p` on disk for future classification use. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
Sliding window search is defined in function `pipeline` in `find_cars.py`. I defined the start point of y axis as 400 while end point as 656. Then I split the search region into 2 sub-windows as [400, 592], [464, 656] respectively. The scaling factor for these 2 sub-windows are 1.5, 2.0 beased on the perspective rule. Also, I did a feature matching for whole search region [400, 656] with scale factor 2.5. Window size is [64, 64] and cells_per_step are set to 2, which means window overlap is 75%. 75% overlap ensures we won't miss any partial car features without sacrificing performance too much.

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Initially I used multiple scales to search on ROI in the image and the speed was very slow. Then I applied different size windows on different regions which yield faster and better result. Here are some example images:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. Then I stored the most recent 5 frames car boxes in `self.car_history` object in `Car.py`. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, the boxes detected is very unstable. It jumped between frames. With the inspiration in the course video and the approach used in Advanced Lane Finding project, I stored historical boxed and created a new heatmap from that for each frame. The historical weight to the current frame adds stability to the detected boxes.

1. Noticed that areas with higher brightness as well as the cars coming from opposite direction gives me most of the false positives.
Add more non-car training data with brightness and largely gradient changes.
2. The detected car boxed are not stable.
Use better average method to smooth detected windows in consecutive frames.
