import pickle, glob
from moviepy.editor import VideoFileClip
from find_cars import *

#test_image_path = '/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/test_images/'
#test_images = [file for file in glob.glob(test_image_path + '*.jpg', recursive=True)]
#for image in test_images:
#	pipeline(image)

# Get lane binary image with color/gradient
white_output = 'test_video_result.mp4'
clip1 = VideoFileClip("/home/linfeng-zc/Documents/Udacity/CarND-Vehicle-Detection/project_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
