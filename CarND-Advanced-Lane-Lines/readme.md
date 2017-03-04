**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/original.jpg "distorted"
[image2]: ./images/undist.jpg "undistorted"
[image3]: ./images/test_image.jpg "Road Transformed"
[image4]: ./images/binary_img.jpg "Binary Example"
[image5]: ./images/warped_lanes.jpg "Warped Example"
[image6]: ./images/color_fit_lines.jpg "Fit Visual"
[image7]: ./images/example_output.jpg "Output"
[video1]: ./project_video_result.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function "calubrate_camera" in "camera_calibration.py" file.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, "objp" is just a replicated array of coordinates, and "objpoints" will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  "imgpoints" will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. The chessboard "imgpoints" are detected using "cv2.findChessboardCorners()"  function with arguments image and chessboard shape.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the "cv2.calibrateCamera()" function.  I applied this distortion correction to the test image using the "cv2.undistort()" function and obtained this result:

![alt text][image1]
![alt text][image2]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I tried to combine gradient and color method together to produce a good binary image, however I could not get the result as good as using color channel alone. Here is my solution in "color_gradient_threshold.py":
1. Crop the image in function "region_of_interest" by only looking at the lower half of the image.
2. Select s channel of the image with threshold in (100, 255] using function "hls_select".
3. Select h channel of the image with threshold in (0, 80] using function "hls_select".
4. Perform a logical AND of the above two results to get the pipline image.

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[511, 511],
     [781, 511],
     [438, 564],
     [859, 564]])

dst = np.float32(
    [[465, 520],
     [815, 520],
     [465, 620],
     [815, 620]])
```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 511, 511      | 465, 520      |
| 781, 511      | 815, 520      |
| 438, 564      | 465, 620      |
| 859, 564      | 815, 620      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The fit_line main logic is in "fit_line_splitter" function in "fit_line.py". For the first frame where no line is detected, it will call the "fit_line" function which is a naive line finding algorithm as defined in the lecture video. The idea of this approach is to
1. Plot the histogram of the binary image.
2. Find the maximum x value in histogram for the left and right half of the image.
3. Given a horizontal and vertical margin, find the x, y value of the points in each  window from bottom to top. Recenter the window to the mean x value of each sliding window when the point in the window is larger0 than 50.
4. Fit the line using np.polyfit() based in the x, y values found in previous step.

Starting from the second frame, "fit_line_splitter" will try to call a faster algorithm "fir_line_quick" function to find the lines rapidly. This algorithm used the coefficients found in the previous frame to find all x, y values within left margin 100 and right margin 100 of the previous found lines. Then it uses np.polyfit() again to find the lines. It is possible that this faster algorithm will not find any lines. In that case, the program will call "fit_line" function to use the naive approach again. And if the difference of the radius of the curvature for the found two lines are larger than 10%, it will also call "fit_line" function.

Calculate the average of previous 5 found coeeficients to smooth the result and treat this "best_fit" as the final result for each frame.

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated in "calculate" function in Line.py. After I calculate the curvatures for left and right lanes, I average them to get the curvature for current frame in fit_line_splitter in fit_line.py.

Position of vehicle is calculated in function calculate_pos in fit_line.py. It first get the middle point of left and right x with y = image.height. Then the offset in pixel level is the difference of the middle point and image center. Multiply the result with 3.7/700 to get real world distance in meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in "process_image" in "main.py" using warp_perspective.  Here is an example of my result on a test image:


![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here is a [link to my video result](./project_video_result.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. To get a good binary image for each frame is a fundemental step of this project. The gradient method works pretty well in most of the frames but it cannot detect the line properly for those images with shadows. Based on a few tests, the satudation channel works pretty well on detecting the lines in shadows. The drawback of only using "s" channel is it can also detects the shadow as well. now the "H" channel comes into play. I specified the hue value between (0, 80] so that it can detect the yellow and some light colors. I took a logical "AND", the result looks better.
2. While finding the line pixels and fitting a polynomial to it, I cannot find any coefficients for some images in the video. In this case, I need to use the average coefficients of the previous 5 frames as a prediction. It works well in most of the cases but if the cuvature of the line is relatively big, the region I found will not cover the correct resion perfectly.
3. However, if the first image cannot find correct lines because of the hazard environment, my algorithm will fail. So it is better to start with an easy frame.
