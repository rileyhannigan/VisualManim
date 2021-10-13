from manim import *
import numpy as np

class Ravines(Scene):
	def ys(self, xs, ws):
		return np.dot(xs, ws)


	def mean_squares_loss(self, ys, ts):
		return np.mean(np.power(ts - ys, 2))


	def normalize(self, xs):
		mean = np.mean(xs, axis=0)
		st_deviation = np.sqrt(np.var(xs, axis=0))
		return (xs - mean)/st_deviation


	"""
	Calculates an Ellipse with the given properties:

	x:	  the x coordinate [Float]
	y:	  the y coordinate [Float]

	xs:	  the input values from the dataset  [narray (n ,2)]
	ts:	  the output values from the dataset [narray (n, 1)]

	radius (float):	  the radius of the ellipse

	Returns a float
	"""
	def ellipse(self, x, y, xs, ts, radius=1):
		return self.mean_squares_loss(self.ys(xs, np.array([x, y])), ts) - radius


	def construct(self):
		r_data = np.array([[2, 30, 5],
				   [4, 30, 7],
				   [8, -80, 0]])

		n_data = np.array([[2, 3, 5],
				   [4, 3, 7],
				   [8, -8, 0]])

		r_dataset = DecimalTable(
			[r_data[0, :], r_data[1, :], r_data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 0}).to_edge(LEFT)


		n_dataset = DecimalTable(
			[n_data[0, :], n_data[1, :], n_data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 0}).to_edge(LEFT)


		# Ellipse math
		r_weights = np.array([[0.874, 0.077],
				      [0.911, 0.084],
				      [0.960, 0.093]])

		r_loss = np.array([
			self.mean_squares_loss(self.ys(r_data[:, :2], r_weights[0, :]), r_data[:, 2]),
			self.mean_squares_loss(self.ys(r_data[:, :2], r_weights[1, :]), r_data[:, 2]),
			self.mean_squares_loss(self.ys(r_data[:, :2], r_weights[2, :]), r_data[:, 2]),
		])

		ravines = [
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=1),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=0.5),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=0.1),
		]

		# Normalized math
		n_weights = np.array([[0.874, 0.770],
				      [0.911, 0.838],
				      [0.960, 0.927]])

		n_loss = np.array([
			self.mean_squares_loss(self.ys(n_data[:, :2], n_weights[0, :]), n_data[:, 2]),
			self.mean_squares_loss(self.ys(n_data[:, :2], n_weights[1, :]), n_data[:, 2]),
			self.mean_squares_loss(self.ys(n_data[:, :2], n_weights[2, :]), n_data[:, 2]),
		])

		normals = [
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=1),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=0.5),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=0.1),
		]

		# Implicit Graphs
		r_graph = Axes(
			x_range=[0.75, 1.25, 0.05],
			y_range=[0.05, 0.15, 0.05],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		n_graph = Axes(
			x_range=[0.75, 1.25, 0.125],
			y_range=[0.75, 1.25, 0.125],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		labels = r_graph.get_axis_labels(x_label='w_1', y_label='w_2')

		# Math Text
		weight_values = [
			MathTex(f"w_1 = {r_weights[0, 0]}, w_2 = {r_weights[0, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[1, 0]}, w_2 = {r_weights[1, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[2, 0]}, w_2 = {r_weights[2, 1]}").next_to(r_dataset, DOWN, buff=1),
		]

		# Formulas
		y_label = MathTex("y = w_1 x_1 + w_2 x_2", color=YELLOW)
		w1_label = MathTex("w_1 \\leftarrow", "w_1 - \\alpha x_1(y - t) ")
		w2_label = MathTex("w_2 \\leftarrow", "w_2 - \\alpha x_2(y - t) ")

		# Calculating Cost
		r_data = np.array([[2, 30, 5],
				   [4, 30, 7],
				   [8, -80, 0]])


		cost_label = MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(w_{1}^{(i)}x_{1}^{(i)} + w_{2}^{(i)}x_{2}^{(i)} - t^{(i)})^2", color=YELLOW).scale(0.75)

		cost1_labels = [
			MathTex("J =", color=RED),
			MathTex("\\frac{1}{3}((2w_{1}^{(0)} + 30w_{2}^{(0)} - 5)^2", color=RED).scale(0.75),
			MathTex("+ (4w_{1}^{(1)} + 30w_{2}^{(1)} - 7)^2", color=RED).scale(0.75),
			MathTex("+ (8w_{1}^{(2)} - 80w_{2}^{(2)} - 0)^2)", color=RED).scale(0.75),
			MathTex("= 1", color=RED),
		]

		dataset_group = Group(r_dataset).to_edge(UP, buff=1).to_edge(LEFT)
		frameboxes = [
			SurroundingRectangle(r_dataset[1], buff=0.3).to_edge(UP, buff=2.4),
			SurroundingRectangle(r_dataset[2], buff=0.3).to_edge(UP, buff=3.5),
			SurroundingRectangle(r_dataset[3], buff=0.3).to_edge(UP, buff=4.7),
		]

		# Creating Scene
		self.add(dataset_group, r_graph)
		self.wait()
		self.play(Create(cost_label.scale(1.15).next_to(r_graph, DOWN)))
		self.wait()

		# First Row Calc
		self.play(Create(cost1_labels[0].move_to(2.75*DOWN + RIGHT/10)))
		self.play(FadeIn(cost1_labels[1].next_to(cost_label, DOWN)))
		self.wait()

		# Second Row Calc
		self.play(FadeIn(cost1_labels[2].next_to(cost1_labels[1], DOWN)))
		self.wait(2)

		# Third Row Calc
		self.play(FadeIn(cost1_labels[3].next_to(cost1_labels[2], DOWN)))
		self.wait(2)

		# Last Row Calc
		self.play(FadeIn(cost1_labels[4].move_to(2.75*DOWN + 5.5*RIGHT)))
		self.wait(2)


		# Show Graph
		r_graphs = [
			r_graph.get_implicit_curve(ravines[0], color=RED),
			r_graph.get_implicit_curve(ravines[1], color=GREEN),
			r_graph.get_implicit_curve(ravines[2], color=BLUE),
		]

		n_graphs = [
			n_graph.get_implicit_curve(normals[0], color=RED),
			n_graph.get_implicit_curve(normals[1], color=GREEN),
			n_graph.get_implicit_curve(normals[2], color=BLUE),
		]

		self.play(Create(r_graph + r_graphs[0]))
		self.wait(2)
