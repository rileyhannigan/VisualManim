from manim import *
import numpy as np

class Ravines(Scene):
	"""
	Calculates an Ellipse with the given properties:

	x (float):	  the x coordinate
	y (float):	  the y coordinate

	x_center (float): the center coordinates,x component
	y_center (float): the center coordinates, y_component

	x_coef (float):	  the coeffecient of x
	y_coef (float):	  the coeffecient of y

	radius (float):	  the radius of the ellipse (shortest radius)

	You can elongate in the x or y direction by manipulating the coeffecients:
		x_coef < y_coef => horizontal ellipse
		x_coef > y_coef => vertical ellipse
		x_coef = y_coef => round ellipse aka circle

	The coeffecients can be used to represent a ratio between coordinates:
		avg_x/avg_y = 4/1 => x_coef = 4, y_coef = 1

	Returns a float
	"""
	def ellipse(self, x, y, x_center=0, y_center=0, x_coef=1, y_coef=1, radius=1):
		return np.power(x - x_center, 2)*x_coef + np.power(y - y_center, 2)*y_coef - radius


	"""
	Calculates the domain of an Ellipse with the given properties:

	x_center (float):	the center x-coordinate for the ellipse
	x_coef	 (float):	the x-coeffecient for the ellipse
	radius   (float):	the radius of the ellipse (shortest radius)

	The domain of the ellipse: [-sqrt(radius*x_coef) + x_center, sqrt(radius*x_coef) + x_center]

	Returns an array that represents the domain of the ellipse given.
	"""
	def ellipse_domain(self, x_center, x_coef, radius):
		inside = radius*x_coef
		return (-1*np.sqrt(inside) + x_center, np.sqrt(inside) + x_center)


	"""
	Calculates the range of an Ellipse with the given properties:

	y_center (float):	the center y-coordinate for the ellipse
	y_coef	 (float):	the y-coeffecient for the ellipse
	radius   (float):	the radius of the ellipse (shortest radius)

	The range of the ellipse: [-sqrt(radius*y_coef) + y_center, sqrt(radius*y_coef) + y_center]

	Returns an array that represents the range of the ellipse given.
	"""
	def ellipse_range(self, y_center, y_coef, radius):
		inside = radius*y_coef
		return (-1*np.sqrt(inside) + y_center + 0.01, np.sqrt(inside) + y_center - 0.01)


	"""
	Calculates the optimal weights for the model given two data points:

	xs:	a numpy array with 2 rows and 2 columns
	ts:	a numpy array with 2 rows and 1 column

	The arrays should be ordered as such:
		xs = [[datapoint1_x1, datapoint1_x2],
		      [datapoint2_x1, datapoint2_x2]]

		ts = [datapoint1_t1,
		      datapoint2_t2]
	"""
	def optimal_weights(self, xs, ts):
		return np.dot(np.linalg.inv(xs), ts)


	"""
	Calculates the new weight value using gradient descent. Expects Mean Squared Loss to be used.

	weight (float): the current value of the weight
	x (float):	the current value of the corresponding x
	y (float):	the current value of the corresponding y
	t (float):	the current value of the corresponding t
	l_rate (float):	the learning rate value
	"""
	def gradient_descent(self, weight, x, y, t, l_rate=0.1):
		return weight - l_rate*2*(y - t)*x


	def construct(self):
		data = np.array([[1, 50, 9],
				 [3, 70, 11],
				 [5, 90, 13]])

		normal_data = np.array([[1, 5, 9],
				 [3, 7, 11],
				 [5, 9, 13]])

		dataset = DecimalTable(
			[data[0, :], data[1, :], data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 0})

			

		normal_dataset = DecimalTable(
			[normal_data[0, :], normal_data[1, :], normal_data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 0})


		init_values=MathTex("w_1 = 0.1, w_2 = 0.2").next_to(dataset, DOWN, buff=1)

		x_values=MathTex("x_1 = 1, x_2 = 50, t = 9").next_to(init_values, DOWN).set_color_by_tex("x_1", YELLOW)


		y_label=MathTex("y =", "w_1 x_1 + w_2 x_2").to_edge(UP, buff=1)
		y_label1=MathTex("y = 0.1 * 1 + 0.2 * 50")
		y_label2=MathTex("y = 10.1")

		w1_label=MathTex("w_1 \\leftarrow", "w_1 - \\alpha x_1(y - t) ").next_to(y_label,DOWN)
		w1_label1=MathTex("w_1 \\leftarrow", "w_1 - \\alpha \\frac{1}{2} (x_1(w_1 x_1 + w_2 x_2 - t)) ").next_to(y_label,DOWN, buff=-0.5)
		w1_label2=MathTex("w_1 \\leftarrow", "0.02 - 0.0005 (x_1(0.02 x_1 + 0.1 x_2 - t)) ").next_to(y_label,DOWN, buff=-0.5)
		w1_label3=MathTex("w_1 \\leftarrow", "0.02 - 0.0005 (114.8(0.02 * 114.8 + 0.1 * 0.00323 - 5.1)) ").next_to(y_label,DOWN, buff=-0.5)
		w1_label4=MathTex("w_1 \\leftarrow", "0.18095").next_to(y_label,DOWN, buff=-0.5)

		w2_label=MathTex("w_2 \\leftarrow", "w_2 - \\alpha x_2(y - t) ").next_to(w1_label,DOWN)
		w2_label1=MathTex("w_2 \\leftarrow", "w_2 - \\alpha \\frac{1}{2} (x_2(w_1 x_1 + w_2 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label2=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (x_2(0.01 x_1 + 0.01 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label3=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (0.00323(0.01 * 114.8 + 0.01 * 0.00323 - 5.1)) ").next_to(w1_label,DOWN)
		w2_label4=MathTex("w_2 \\leftarrow", "0.010004 ").next_to(w1_label,DOWN)

		loss_label=MathTex("L =\\frac{1}{2} (y - t)^2").next_to(w2_label, DOWN)

		cost_label=MathTex("J=\\frac{1}{2N}\\sum_{i=1}^{N}(y^{(i)} - t^{(i)})^2").next_to(w2_label, DOWN)

		# Ellipse math
		ws = self.optimal_weights(data[:2, :2], data[:2, 2])
		x_coef = np.abs(ws[0]/ws[1])
		y_coef = 1

		r1 = 1
		r2 = 0.5
		r3 = 0.1

		e1 = lambda x, y: self.ellipse(x, y, x_center=ws[0], y_center=ws[1], x_coef=x_coef, y_coef=y_coef, radius=r1)
		e2 = lambda x, y: self.ellipse(x, y, x_center=ws[0], y_center=ws[1], x_coef=x_coef, y_coef=y_coef, radius=r2)
		e3 = lambda x, y: self.ellipse(x, y, x_center=ws[0], y_center=ws[1], x_coef=x_coef, y_coef=y_coef, radius=r3)

		e1_domain = self.ellipse_domain(x_center=ws[0], x_coef=x_coef, radius=r1)
		e2_domain = self.ellipse_domain(x_center=ws[0], x_coef=x_coef, radius=r2)
		e3_domain = self.ellipse_domain(x_center=ws[0], x_coef=x_coef, radius=r3)

		e1_range = self.ellipse_range(y_center=ws[1], y_coef=y_coef, radius=r1)
		e2_range = self.ellipse_range(y_center=ws[1], y_coef=y_coef, radius=r2)
		e3_range = self.ellipse_range(y_center=ws[1], y_coef=y_coef, radius=r3)


		# Normalized math
		nws = self.optimal_weights(normal_data[:2, :2], normal_data[:2, 2])
		normal_x_coef = np.abs(nws[0]/nws[1])
		normal_y_coef = 1

		n1 = lambda x, y: self.ellipse(x, y, x_center=nws[0], y_center=nws[1], x_coef=normal_x_coef, y_coef=normal_y_coef, radius=r1)
		n2 = lambda x, y: self.ellipse(x, y, x_center=nws[0], y_center=nws[1], x_coef=normal_x_coef, y_coef=normal_y_coef, radius=r2)
		n3 = lambda x, y: self.ellipse(x, y, x_center=nws[0], y_center=nws[1], x_coef=normal_x_coef, y_coef=normal_y_coef, radius=r3)

		n1_domain = self.ellipse_domain(x_center=nws[0], x_coef=normal_x_coef, radius=r1)
		n2_domain = self.ellipse_domain(x_center=nws[0], x_coef=normal_x_coef, radius=r2)
		n3_domain = self.ellipse_domain(x_center=nws[0], x_coef=normal_x_coef, radius=r3)

		n1_range = self.ellipse_range(y_center=nws[1], y_coef=normal_y_coef, radius=r1)
		n2_range = self.ellipse_range(y_center=nws[1], y_coef=normal_y_coef, radius=r2)
		n3_range = self.ellipse_range(y_center=nws[1], y_coef=normal_y_coef, radius=r3)

		# Implicit Graph

		misc_label=MathTex("w_1 = 0.02, w_2 = 0.01, \\alpha = 0.001")

		graph = Axes(
			x_range=[e1_domain[0] - 0.5, e1_domain[1] + 0.5, (e1_domain[1] - e1_domain[0])/4],
			y_range=[e1_range[0] - 0.5, e1_range[1] + 0.5, (e1_range[1] - e1_range[0])/4],
			x_length = 8,
			y_length = 8,
			axis_config={"include_numbers": False}
		)

		normal_graph = Axes(
			x_range=[n1_domain[0] - 0.5, n1_domain[1] + 0.5, (n1_domain[1] - n1_domain[0])/4],
			y_range=[n1_range[0] - 0.5, n1_range[1] + 0.5, (n1_range[1] - n1_range[0])/4],
			x_length = 8,
			y_length = 8,
			axis_config={"include_numbers": False}
		)

		labels = graph.get_axis_labels(x_label='w_1', y_label='w_2')

		e1_graph = graph.get_implicit_curve(e1, color=RED)
		e2_graph = graph.get_implicit_curve(e2,  color=GREEN)
		e3_graph = graph.get_implicit_curve(e3, color=BLUE)

		n1_graph = normal_graph.get_implicit_curve(n1, color=RED)
		n2_graph = normal_graph.get_implicit_curve(n2,  color=GREEN)
		n3_graph = normal_graph.get_implicit_curve(n3, color=BLUE)

		graph_group = Group(graph, labels, e1_graph, e2_graph, e3_graph).scale(0.8).to_edge(RIGHT)
		normal_graph_group = Group(normal_graph, labels, n1_graph, n2_graph, n3_graph).scale(0.6)
		dataset_group = Group(dataset, init_values).to_edge(UP, buff=1).to_edge(LEFT)
		formula_group = Group(cost_label, y_label, w1_label, w2_label).to_edge(RIGHT)
		#group1 = Group(dataset_group, graph_group).arrange(buff=1)
		framebox = SurroundingRectangle(dataset[1], buff=0.3).to_edge(UP, buff=2.4)	
		x_values.next_to(init_values, DOWN)
		y_label1.move_to(y_label).to_edge(LEFT, buff=2)
		y_label2.move_to(y_label).to_edge(LEFT, buff=2)



		# Creating Scene
		self.add(dataset_group)
		self.add(formula_group)
		#self.add(init_values)
		#highlight around top row
		# #self.add(cost_label)
		self.wait()
		self.play(Create(framebox))
		self.wait()
		self.play(Create(x_values))
		self.wait()
		self.play(FadeOut(dataset), FadeOut(framebox))
		self.wait()
		self.play(formula_group.animate.shift(LEFT * 8.5))
		self.wait()
		self.play(FadeIn(graph_group))
		self.wait(3)


		# self.play(FadeIn(y_label1),FadeOut(y_label))
		# self.wait()
		# self.play(FadeIn(y_label2),FadeOut(y_label1))
		# self.wait()
		# self.wait()
		# self.play(Transform(graph_group, normal_graph_group))

		#self.play(Transform(y_label, y_label1))
		# self.play(Transform(w2_label, w2_label1))
		# self.wait()
		# self.play(Transform(w1_label, w1_label2))
		# self.play(Transform(w2_label, w2_label2))
		# self.wait()
		# self.play(Transform(w1_label, w1_label3))
		# self.play(Transform(w2_label, w2_label3))
		# self.wait()
		# self.play(Transform(w1_label, w1_label4))
		# self.play(Transform(w2_label, w2_label4))
