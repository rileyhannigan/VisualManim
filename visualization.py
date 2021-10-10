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
		)

		n_graph = Axes(
			x_range=[0.75, 1.25, 0.125],
			y_range=[0.75, 1.25, 0.125],
			axis_config={"include_numbers": False}
		)

		labels = r_graph.get_axis_labels(x_label='w_1', y_label='w_2')

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

		# Math Text
		weight_values = [
			MathTex(f"w_1 = {r_weights[0, 0]}, w_2 = {r_weights[0, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[1, 0]}, w_2 = {r_weights[1, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[2, 0]}, w_2 = {r_weights[2, 1]}").next_to(r_dataset, DOWN, buff=1),
		]

		# Formulas
		y_label = MathTex("y = w_1 x_1 + w_2 x_2", color=YELLOW)
		cost_label = MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(y_i} - t_i)^2", color=RED)
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
			MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(y_i - t_i)^2", color=RED),
			MathTex("J = \\frac{1}{3}\\sum_{i=1}^{3}(y_i - t_i)^2", color=RED),
			MathTex("J = \\frac{1}{3} (y_1 - t_1)^2 + (y_2 - t_2)^2 + (y_3 - t_3)^2", color=RED),
			MathTex(f"J = {round(r_loss[0], 3)}", color=RED),
		]

		# Grouping
		r_graph_groups = [
			Group(r_graph, labels).scale(1).to_edge(RIGHT).to_edge(UP),
			Group(r_graph, labels, r_graphs[0]).scale(7).to_edge(RIGHT).to_edge(UP),
			Group(r_graph, labels, r_graphs[0], r_graphs[1]).scale(7).to_edge(RIGHT).to_edge(UP),
			Group(r_graph, labels, r_graphs[0], r_graphs[1], r_graphs[2]).scale(7).to_edge(RIGHT).to_edge(UP),
		]

		n_graph_groups = [
			Group(n_graph, labels).scale(7).to_edge(RIGHT).to_edge(UP),
			Group(n_graph, labels, n_graphs[0]).scale(7).to_edge(RIGHT).to_edge(UP),
			Group(n_graph, labels, n_graphs[0], n_graphs[1]).scale(7).to_edge(RIGHT).to_edge(UP),
			Group(n_graph, labels, n_graphs[0], n_graphs[1], n_graphs[2]).scale(7).to_edge(RIGHT).to_edge(UP),
		]

		dataset_group = Group(r_dataset, weight_values[0]).to_edge(UP, buff=1).to_edge(LEFT)
		frameboxes = [
			SurroundingRectangle(r_dataset[1], buff=0.3).to_edge(UP, buff=2.4),
			SurroundingRectangle(r_dataset[2], buff=0.3).to_edge(UP, buff=3.5),
			SurroundingRectangle(r_dataset[3], buff=0.3).to_edge(UP, buff=4.7),
		]

		# Creating Scene
		self.add(dataset_group, r_graph.scale(1.25))
		self.wait()
		self.play(Create(y_label.scale(1.25)))
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
		self.play(FadeIn(cost_label.next_to(y3_labels[1], DOWN)))
		self.wait(2)
		self.play(Transform(cost_label, cost_labels[0].next_to(y3_labels[1], DOWN)))

		# Cost Calc 2
		self.wait(2)
		self.play(Transform(cost_labels[0], cost_labels[1].next_to(y3_labels[1], DOWN)))

		# Cost Calc 3
		self.wait(2)
		self.play(Transform(cost_labels[1], cost_labels[1].next_to(y3_labels[2], DOWN)))

		# Cost Calc 4
		self.wait(2)
		self.play(Transform(cost_labels[2], cost_labels[1].next_to(y3_labels[3], DOWN)))

		# Show Graph
#		self.play(Transform(r_graph_groups[0], r_graph_groups[1]))
