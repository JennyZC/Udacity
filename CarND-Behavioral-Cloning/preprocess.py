import cv2
import numpy as np

ORG_ROW = 160
ORG_COL = 320

ROI_ROW_START = 60
ROI_ROW_END = 135
ROI_COL_START = 0
ROI_COL_END = ORG_COL

RESIZE_FACTOR = 5

ROWS = round((ROI_ROW_END - ROI_ROW_START) / RESIZE_FACTOR)
COLS = round((ROI_COL_END - ROI_COL_START) / RESIZE_FACTOR)
print("roi:", ROI_COL_START, ROI_COL_END, "row: ", ROWS, "Cols: ", COLS)
CHANNELS = 1

# Convert one channel 2D image to 3D image with dimention (rows, cols, 1)
def to_rank3(image):
	if len(image.shape) == 3:
		return image
	
	rank3_image = np.zeros((image.shape[0], image.shape[1], 1), dtype=image.dtype)
	rank3_image[:, :, 0] = image
	return rank3_image

def flip_img(img):
	flipped_img = cv2.flip(img, 1)
	return to_rank3(flipped_img)

# Preprocessing images
def preprocess(image):
	# To YUV
	img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

	# Crop ROI and take Y channel 
	img_roi = img_yuv[ROI_ROW_START:ROI_ROW_END, ROI_COL_START:ROI_COL_END, 0]
		
	# Resize image
	resized_roi = cv2.resize(img_roi, (COLS, ROWS))

	return to_rank3(resized_roi)

def process_line(line):
	img_center = cv2.imread(line[0])
	img_left = cv2.imread(line[1])
	img_right = cv2.imread(line[2])
	correction = 0.03
	return [(img_center, float(line[3])), (img_left, float(line[3]) + correction), (img_right, float(line[3]) - correction)]
	#return [(img_center, float(line[3]))]

if __name__ == '__main__':
	pass
