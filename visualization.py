from manim import *
import numpy as np

class Ravines(Scene):
	def ys(self, xs, ws):
		return np.dot(xs, ws)


	def mean_squares_loss(self, ys, ts):
		return np.mean(np.power(ts - ys, 2))


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

		n_data = np.array([[-1.1, 0.7, 5],
				   [-0.3, 0.7, 7],
				   [1.3, 1.7, 0]])

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
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=r_loss[0]),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=r_loss[1]),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=r_loss[2]),
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
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=n_loss[0]),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=n_loss[1]),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=n_loss[2]),
		]

		# Implicit Graphs
		r_graph = Axes(
			x_range=[0.75, 1.25, 0.2],
			y_range=[0.05, 0.15, 0.05],
			axis_config={"include_numbers": False}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		n_graph = Axes(
			x_range=[0.75, 1.25, 0.125],
			y_range=[0.75, 1.25, 0.125],
			axis_config={"include_numbers": False}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		labels = n_graph.get_axis_labels(x_label='w_1', y_label='w_2')

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

		# Calculating Ys
		y1_labels = [
			MathTex(f"y_1 = {r_weights[0, 0]} * {r_data[0, 0]} + {r_weights[0, 1]} * {r_data[0, 1]}"),
			MathTex(f"y_1 = {round(r_weights[0, 0] * r_data[0, 0] + r_weights[0, 1] * r_data[0, 1], 3)}"),
		]

		y2_labels = [
			MathTex(f"y_2 = {r_weights[0, 0]} * {r_data[1, 0]} + {r_weights[0, 1]} * {r_data[1, 1]}"),
			MathTex(f"y_2 = {round(r_weights[0, 0] * r_data[1, 0] + r_weights[0, 1] * r_data[1, 1], 3)}"),
		]

		y3_labels = [
			MathTex(f"y_3 = {r_weights[0, 0]} * {r_data[2, 0]} + {r_weights[0, 1]} * {r_data[2, 1]}"),
			MathTex(f"y_3 = {round(r_weights[0, 0] * r_data[2, 0] + r_weights[0, 1] * r_data[2, 1], 3)}"),
		]

		# Calculating Cost
		cost_labels = [
			MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - t_i)^2", color=RED).scale(0.8),
			MathTex("J = \\frac{1}{3}\\sum_{i=1}^{3}(y_i - t_i)^2", color=RED).scale(0.8),
			MathTex("J = \\frac{1}{3} [(y_1 - t_1)^2 + (y_2 - t_2)^2 + (y_3 - t_3)^2]", color=RED).scale(0.8),
			MathTex(f"J = {round(r_loss[0], 3)}", color=RED),
		]

		dataset_group = Group(n_dataset, weight_values[0]).to_edge(UP, buff=1).to_edge(LEFT)
		frameboxes = [
			SurroundingRectangle(n_dataset[1], buff=0.3).to_edge(UP, buff=2.4),
			SurroundingRectangle(n_dataset[2], buff=0.3).to_edge(UP, buff=3.5),
			SurroundingRectangle(n_dataset[3], buff=0.3).to_edge(UP, buff=4.7),
		]

		# Creating Scene
		self.add(dataset_group, n_graph)
		self.wait()
		self.play(Create(y_label.scale(1.15).next_to(n_graph, DOWN)))
		self.wait()

		# First Row Calc
		self.play(Create(frameboxes[0]))
		self.wait()

		self.play(FadeIn(y1_labels[0].next_to(y_label, DOWN)))
		self.wait(2)
		self.play(Transform(y1_labels[0], y1_labels[1].next_to(y_label, DOWN)))
		self.wait()

		# Second Row Calc
		self.play(FadeOut(frameboxes[0]))
		self.play(Create(frameboxes[1]))
		self.wait()

		self.play(FadeIn(y2_labels[0].next_to(y1_labels[1], DOWN)))
		self.wait(2)
		self.play(Transform(y2_labels[0], y2_labels[1].next_to(y1_labels[1], DOWN)))
		self.wait()

		# Third Row Calc
		self.play(FadeOut(frameboxes[1]))
		self.play(Create(frameboxes[2]))
		self.wait()

		self.play(FadeIn(y3_labels[0].next_to(y2_labels[1], DOWN)))
		self.wait(2)
		self.play(Transform(y3_labels[0], y3_labels[1].next_to(y2_labels[1], DOWN)))
		self.wait()

		# Cost Calc 1
		self.play(Create(cost_labels[0].next_to(y3_labels[1], DOWN)))
		self.wait(2)

		# Cost Calc 2
		self.play(ReplacementTransform(cost_labels[0], cost_labels[1].next_to(y3_labels[1], DOWN)))
		self.wait(2)

		# Cost Calc 3
		self.play(ReplacementTransform(cost_labels[1], cost_labels[2].next_to(y3_labels[1], DOWN)))
		self.wait(4)

		# Cost Calc 4
		self.play(ReplacementTransform(cost_labels[2], cost_labels[3].next_to(y3_labels[1], DOWN)))
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
