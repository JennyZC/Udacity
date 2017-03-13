class Car():
	def __init__(self, history_len):
		self.car_history = []
		self.history_len = history_len

	def add_car(self, box_list):
		self.car_history.append(box_list)
		if len(self.car_history) > self.history_len:
			self.car_history.pop(0) 
