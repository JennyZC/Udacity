import numpy as np

class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False
		# x values of the last n fits of the line
		self.recent_xfitted = []
		# current x values of the fitted line over the last n iterations
		self.currentx = None
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

	def calculate(self, ploty):
		y_eval = np.max(ploty)

		# Define conversions in x and y from pixels space to meters
		ym_per_pix = 30/720 # meters per pixel in y dimension
		xm_per_pix = 3.7/700 # meters per pixel in x dimension

		# Fit new polynomials to x,y in world space
		fit_cr = np.polyfit(ploty*ym_per_pix, self.currentx*xm_per_pix, 2)

		# Calculate the new radii of curvature
		curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

		# Now our radius of curvature is in meters
		print("Radius of Curvature: ", curverad, 'm')
		# Example values: 632.1 m    626.2 m

		self.radius_of_curvature = curverad
