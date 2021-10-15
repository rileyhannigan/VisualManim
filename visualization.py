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

		n_data = np.array([[-1.07, 0.70, 5],
				   [-0.27, 0.70, 7],
				   [1.34, -1.41, 0]])

		r_dataset = DecimalTable(
			[r_data[0, :], r_data[1, :], r_data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 0}).to_edge(LEFT)


		n_dataset = DecimalTable(
			[n_data[0, :], n_data[1, :], n_data[2, :]],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 0.4,
			element_to_mobject_config={"num_decimal_places": 2}).to_edge(LEFT).to_edge(UP, buff=1)


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

		# Normalizing Formulas
		mean_formula = MathTex(r"\mu_i = \frac{\sum_j^{m}x_{i,j}}{m}").shift(UP*3,RIGHT*2)
		variance_formula = MathTex(r"\sigma_i^2 = \frac{\sum_j{(x_{i,j}-\mu_i)^2}}{m}").shift(UP*1.5, RIGHT*2)
		
		# mu labels
		mu_1_formula = MathTex(r"\mu_1 = \frac{\sum_j^{m}x_{1,j}}{m}").shift(mean_formula.get_center())
		mu_2_formula = MathTex(r"\mu_2 = \frac{\sum_j^{m}x_{2,j}}{m}").shift(mean_formula.get_center(), DOWN*1.5)
		mu_1_expanded = MathTex(r"\mu_1 = \frac{2+4+8}{3}=4.67").shift(mu_1_formula.get_center())
		mu_2_expanded = MathTex(r"\mu_2 = \frac{30+30-80}{3}=-6.67").shift(mu_2_formula.get_center())
		mu_1 = MathTex(r'\mu_1 = 4.67').shift(mu_1_expanded.get_center(), LEFT*1.5)
		mu_2 = MathTex(r'\mu_2 = -6.67').shift(mu_2_expanded.get_center(), LEFT*1.5 )

		# variance labels
		var_1_formula = MathTex(r"\sigma_1 = \frac{\sum_j{(x_{1,j}-\mu_1)^2}}{m}").shift(variance_formula.get_center(), DOWN*2)
		var_2_formula = MathTex(r"\sigma_2 = \frac{\sum_j{(x_{2,j}-\mu_2)^2}}{m}").shift(variance_formula.get_center(), DOWN*4)
		var_1_expanded = MathTex(r"\sigma_1 = \frac{(2-4.67)^2 + (4-4.67)^2 + (8-4.67)^2}{3} = 6.22").scale(0.6).shift(var_1_formula.get_center(), RIGHT)
		var_1 = MathTex(r"\sigma_1=6.22").shift(mu_2.get_center(), DOWN*1.5)
		var_2_expanded = MathTex(r"\sigma_2 = \frac{(30+6.67)^2 + (30+6.67)^2 + (-80+6.67)^2}{3} = 2688.89").scale(0.6).shift(var_2_formula.get_center(), RIGHT)
		var_2 = MathTex(r"\sigma_2=2688.89").shift(var_1.get_center(), DOWN*1.5)

		# norm labels
		x_1_norm = MathTex(r"\hat{x_1} = \frac{x_1 - \mu_1}{\sqrt{\sigma_1}}").shift(mu_1.get_center(), RIGHT*3)
		x_2_norm = MathTex(r"\hat{x_2} = \frac{x_2 - \mu_2}{\sqrt{\sigma_2}}").shift(mu_2.get_center(), RIGHT*3)
		x_1_norm_sub = MathTex(r"\hat{x_1} = \frac{x_1 - 4.67}{\sqrt{6.22}}=").shift(mu_1.get_center(), RIGHT*3.5)
		x_2_norm_sub = MathTex(r"\hat{x_2} = \frac{x_2 + 6.67}{\sqrt{2688.89}}=").shift(mu_2.get_center(), RIGHT*3.5)
		x_1_1 = MathTex('-1.07').shift(x_1_norm.get_right(), LEFT*0.3)
		x_1_2 = MathTex('0.7').shift(x_2_norm.get_right(), LEFT*0.3)
		x_2_1 = MathTex('-0.27').shift(x_1_norm.get_right(), LEFT*0.3)
		x_2_2 = MathTex('0.7').shift(x_2_norm.get_right(), LEFT*0.3)
		x_3_1 = MathTex('1.34').shift(x_1_norm.get_right(), LEFT*0.3)
		x_3_2 = MathTex('-1.41').shift(x_2_norm.get_right(), LEFT*0.3)

		# Implicit Graphs
		r_graph = Axes(
			x_range=[0.75, 1.25, 0.05],
			y_range=[0.05, 0.15, 0.05],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		n_graph = Axes(
			x_range=[0.0, 1.25, 0.125],
			y_range=[0.0, 1.25, 0.125],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		labels = r_graph.get_axis_labels(x_label='w_1', y_label='w_2')

		# Math Text
		weight_values = [
			MathTex(f"w_1 = {r_weights[0, 0]}, w_2 = {r_weights[0, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[1, 0]}, w_2 = {r_weights[1, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[2, 0]}, w_2 = {r_weights[2, 1]}").next_to(r_dataset, DOWN, buff=1),
		]

		# Graphing Formulas
		y_label = MathTex("y = w_1 x_1 + w_2 x_2", color=YELLOW)
		w1_label = MathTex("w_1 \\leftarrow", "w_1 - \\alpha x_1(y - t) ")
		w2_label = MathTex("w_2 \\leftarrow", "w_2 - \\alpha x_2(y - t) ")

		# Calculating Cost
		
		cost_label = MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(w_{1}^{(i)}x_{1}^{(i)} + w_{2}^{(i)}x_{2}^{(i)} - t^{(i)})^2", color=YELLOW).scale(0.75)

		cost1_labels = [
			MathTex("J =", color=RED),
			MathTex("\\frac{1}{3}((-1.07w_{1}^{(0)} + 0.7w_{2}^{(0)} - 5)^2", color=RED).scale(0.75),
			MathTex("+ (-0.27w_{1}^{(1)} + 0.7w_{2}^{(1)} - 7)^2", color=RED).scale(0.75),
			MathTex("+ (1.34w_{1}^{(2)} - 1.41w_{2}^{(2)} - 0)^2)", color=RED).scale(0.75),
			MathTex("= 1", color=RED),
		]

		dataset_group = Group(r_dataset).to_edge(UP, buff=1).to_edge(LEFT)
		frameboxes = [
			SurroundingRectangle(r_dataset[1], buff=0.3).to_edge(UP, buff=2.4),
			SurroundingRectangle(r_dataset[2], buff=0.3).to_edge(UP, buff=3.5),
			SurroundingRectangle(r_dataset[3], buff=0.3).to_edge(UP, buff=4.7),
		]

		# Creating Scene
		self.add(dataset_group)

		# Normalize
		self.play(Write(mu_1_formula), 
                  Write(mu_2_formula),
                  Write(var_1_formula),
                  Write(var_2_formula))
		self.wait()

		# Getting Mus
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_columns()[0])))
		self.play(Transform(mu_1_formula, mu_1_expanded))
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_columns()[1])))
		self.play(Transform(mu_2_formula, mu_2_expanded))
		self.wait()
		self.play(Transform(mu_1_formula, mu_1), Transform(mu_2_formula, mu_2))
		self.wait()

		# Getting Variances
		self.play(Indicate(var_1_formula))
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_columns()[0])))
		self.play(Circumscribe(SurroundingRectangle(mu_1)))
		self.play(Transform(var_1_formula, var_1_expanded))
		self.wait()
		self.play(Transform(var_1_formula, var_1))
		self.wait()
		self.play(Indicate(var_2_formula))
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_columns()[1])))
		self.play(Circumscribe(SurroundingRectangle(mu_2)))
		self.play(Transform(var_2_formula, var_2_expanded))
		self.wait()
		self.play(Transform(var_2_formula, var_2))
		self.wait()

		# Getting Norms
		self.play(Write(x_1_norm), Write(x_2_norm))
		self.play(Transform(x_1_norm, x_1_norm_sub))
		self.play(Transform(x_2_norm, x_2_norm_sub))
		self.play(FadeOut(mu_1_formula), FadeOut(mu_2_formula), FadeOut(var_1_formula), FadeOut(var_2_formula))
		self.play(x_1_norm.animate.shift(LEFT*2), x_2_norm.animate.shift(LEFT*2))
		self.wait()
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_rows()[1])))
		self.play(Write(x_1_1), Write(x_1_2))
		self.wait()
		self.play(Transform(r_dataset.get_rows()[1][0], n_dataset.get_rows()[1][0]),
			Transform(r_dataset.get_rows()[1][1], n_dataset.get_rows()[1][1]))
		self.wait()
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_rows()[2])))
		self.play(Transform(x_1_1, x_2_1), Transform(x_1_2, x_2_2))
		self.wait()
		self.play(Transform(r_dataset.get_rows()[2][0], n_dataset.get_rows()[2][0]),
			Transform(r_dataset.get_rows()[2][1], n_dataset.get_rows()[2][1]))
		self.wait()
		self.play(Circumscribe(SurroundingRectangle(r_dataset.get_rows()[3])))
		self.play(Transform(x_1_1, x_3_1), Transform(x_1_2, x_3_2))
		self.wait()
		self.play(Transform(r_dataset.get_rows()[3][0], n_dataset.get_rows()[3][0]),
			Transform(r_dataset.get_rows()[3][1], n_dataset.get_rows()[3][1]))
		self.wait()
		self.play(FadeOut(x_1_norm), FadeOut(x_2_norm), FadeOut(x_1_1), FadeOut(x_1_2))

		self.add(n_graph)
		self.wait()
		self.play(Create(cost_label.scale(1.15).next_to(n_graph, DOWN)))
		self.wait()

		# First Row Calc
		self.play(Create(cost1_labels[0].move_to(2.75*DOWN + RIGHT/10)))
		self.play(FadeIn(cost1_labels[1].next_to(cost_label, DOWN)))
		self.wait()

		# Second Row Calc
		self.play(FadeIn(cost1_labels[2].next_to(cost1_labels[1], DOWN)))
		self.wait()

		# Third Row Calc
		self.play(FadeIn(cost1_labels[3].next_to(cost1_labels[2], DOWN)))
		self.wait()

		# Last Row Calc
		self.play(FadeIn(cost1_labels[4].move_to(2.75*DOWN + 5.5*RIGHT)))
		self.wait()


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

		self.play(Create(r_graphs[0]))
		self.play(Transform(r_graphs[0], n_graphs[0]))
	# self.play(Create(n_graphs[2]))
		self.wait(2)
