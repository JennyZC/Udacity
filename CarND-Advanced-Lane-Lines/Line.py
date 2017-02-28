import numpy as np

class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		# average x values of the fitted line over the last n iterations
		self.bestx = None
		# polynomial coefficients averaged over the last n iterations
		self.best_fit = None
		# polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]
		# polynomial coefficients for the last n fit
		self.recent_fit = []
		# radius of curvature of the line in some units
		self.radius_of_curvature = None
		# distance in meters of vehicle center from the line
		self.line_base_pos = None
		# difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float')
		# x values for detected line pixels
		self.allx = None
		# y values for detected line pixels
		self.ally = None

	def calculate(self, ploty, current_fit):
		y_eval = np.max(ploty)
		curverad = ((1 + (2*current_fit[0]*y_eval + current_fit[1])**2)**1.5) / np.absolute(2*current_fit[0])
		#print("Radius of Curvature: ", curverad)
		self.radius_of_curvature = curverad
