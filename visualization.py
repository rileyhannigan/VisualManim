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


		n_data = np.ones((3, 3))
		n_data[:, :2] = np.round(self.normalize(r_data[:, :2]), 2)
		n_data[:, 2] = r_data[:, 2]

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

		ravines = [
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=1),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=0.8),
				lambda x, y: self.ellipse(x, y, r_data[:, :2], r_data[:, 2], radius=0.65),
		]

		# Normalized math
		n_weights = np.array([[0.874, 0.077],
				      [0.911, 0.084],
				      [0.960, 0.093]])

		normals = [
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=19),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=18),
				lambda x, y: self.ellipse(x, y, n_data[:, :2], n_data[:, 2], radius=17),
		]

		# Normalizing Formulas
		mean_formula = MathTex(r"\mu_i = \frac{\sum_j^{N}x_{i,j}}{N}").shift(UP*3,RIGHT*2)
		variance_formula = MathTex(r"\sigma_i^2 = \frac{\sum_j{(x_{i,j}-\mu_i)^2}}{m}").shift(UP*1.5, RIGHT*2)

		# mu labels
		mu_1_formula = MathTex(r"\mu_1 = \frac{\sum_j^{N}x_{1,j}}{N}").shift(mean_formula.get_center())
		mu_2_formula = MathTex(r"\mu_2 = \frac{\sum_j^{N}x_{2,j}}{N}").shift(mean_formula.get_center(), DOWN*1.5)
		mu_1_expanded = MathTex(r"\mu_1 = \frac{2+4+8}{3}=4.67").shift(mu_1_formula.get_center())
		mu_2_expanded = MathTex(r"\mu_2 = \frac{40+80-160}{3}=-13.33").shift(mu_2_formula.get_center())
		mu_1 = MathTex(r'\mu_1 = 4.67').shift(mu_1_expanded.get_center(), LEFT*1.5)
		mu_2 = MathTex(r'\mu_2 = -13.33').shift(mu_2_expanded.get_center(), LEFT*1.5 )

		# variance labels 
		## ADD SQUARE ROOT 
		var_1_formula = MathTex(r"\sigma_1 = \sqrt{\frac{\sum_j{(x_{1,j}-\mu_1)^2}}{N}}").shift(variance_formula.get_center(), DOWN*2)
		var_2_formula = MathTex(r"\sigma_2 = \sqrt{\frac{\sum_j{(x_{2,j}-\mu_2)^2}}{N}}").shift(variance_formula.get_center(), DOWN*4)
		var_1_expanded = MathTex(r"\sigma_1 = \sqrt{\frac{(2-4.67)^2 + (4-4.67)^2 + (8-4.67)^2}{3}} = \sqrt{6.22}").scale(0.6).shift(var_1_formula.get_center(), RIGHT)
		var_1 = MathTex(r"\sigma_1=2.49").shift(mu_2.get_center(), DOWN*1.5)
		var_2_expanded = MathTex(r"\sigma_2 = \frac{(40+6.67)^2 + (80+6.67)^2 + (-160+6.67)^2}{3} = \sqrt{11022.22}").scale(0.6).shift(var_2_formula.get_center(), RIGHT)
		var_2 = MathTex(r"\sigma_2=104.99").shift(var_1.get_center(), DOWN*1.5)

		# norm labels
		x_1_norm = MathTex(r"\hat{x_1} = \frac{x_1 - \mu_1}{\sigma_1}").shift(mu_1.get_center(), RIGHT*3)
		x_2_norm = MathTex(r"\hat{x_2} = \frac{x_2 - \mu_2}{\sigma_2}").shift(mu_2.get_center(), RIGHT*3)
		x_1_norm_sub = MathTex(r"\hat{x_1} = \frac{x_1 - 4.67}{2.48}=").shift(mu_1.get_center(), RIGHT*3.5)
		x_2_norm_sub = MathTex(r"\hat{x_2} = \frac{x_2 + 13.33}{104.99}=").shift(mu_2.get_center(), RIGHT*3.5)
		x_1_1 = MathTex('-1.07').shift(x_1_norm.get_right(), LEFT*0.1)
		x_1_2 = MathTex('0.51').shift(x_2_norm.get_right(), LEFT*0.1)
		x_2_1 = MathTex('-0.27').shift(x_1_norm.get_right(), LEFT*0.1)
		x_2_2 = MathTex('0.89').shift(x_2_norm.get_right(), LEFT*0.1)
		x_3_1 = MathTex('1.34').shift(x_1_norm.get_right(), LEFT*0.1)
		x_3_2 = MathTex('-1.40').shift(x_2_norm.get_right(), LEFT*0.1)

		# Implicit Graphs
		r_graph = Axes(
			x_range=[0.8, 1.1, 0.075],
			y_range=[0, 0.16, 0.075],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		n_graph = Axes(
			x_range=[-5.25, 6.75, 1.875],
			y_range=[-0.2, 7.4, 1.875],
			axis_config={"include_numbers": True}
		).scale(0.5).to_edge(RIGHT, buff=1).to_edge(UP)

		labels = r_graph.get_axis_labels(x_label='w_1', y_label='w_2')

		# Math Text
		weight_values = [
			MathTex(f"w_1 = {r_weights[0, 0]}, w_2 = {r_weights[0, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[1, 0]}, w_2 = {r_weights[1, 1]}").next_to(r_dataset, DOWN, buff=1),
			MathTex(f"w_1 = {r_weights[2, 0]}, w_2 = {r_weights[2, 1]}").next_to(r_dataset, DOWN, buff=1),
		]

		# Calculating Cost
		cost_label = MathTex("J = \\frac{1}{N}\\sum_{i=1}^{N}(w_{1}^{(i)}x_{1}^{(i)} + w_{2}^{(i)}x_{2}^{(i)} - t^{(i)})^2", color=ORANGE).scale(0.75)

		costr_labels = [
			MathTex("J = \\frac{1}{3}"),
			MathTex("(2w_{1}^{(0)} + 40w_{2}^{(0)} - 5)^2 +").scale(0.75),
			MathTex("(4w_{1}^{(1)} + 80w_{2}^{(1)} - 7)^2 +").scale(0.75),
			MathTex("(8w_{1}^{(2)} - 1600w_{2}^{(2)} - 0)^2  ").scale(0.75),
		]

		costr_radius = [
			MathTex("= 1",  color=RED).scale(1.15),
			MathTex("= 0.8", color=GREEN).scale(1.15),
			MathTex("= 0.65", color=BLUE).scale(1.15),
		]

		costn_labels = [
			MathTex("J = \\frac{1}{3}"),
			MathTex("(-1.07w_{1}^{(0)} - 0.51w_{2}^{(0)} - 5)^2 +").scale(0.75),
			MathTex("(-0.27w_{1}^{(1)} + 0.89w_{2}^{(1)} - 7)^2 +").scale(0.75),
			MathTex("(1.34w_{1}^{(2)} - 1.40w_{2}^{(2)} - 0)^2  ").scale(0.75),
		]

		costn_radius = [
			MathTex("= 19", color=RED).scale(1.15),
			MathTex("= 18", color=GREEN).scale(1.15),
			MathTex("= 17", color=BLUE).scale(1.15),
		]

		dataset_group = Group(r_dataset).to_edge(UP, buff=1).to_edge(LEFT)
		frameboxes = [
			SurroundingRectangle(r_dataset[1], buff=0.3).to_edge(UP, buff=2.4),
			SurroundingRectangle(r_dataset[2], buff=0.3).to_edge(UP, buff=3.5),
			SurroundingRectangle(r_dataset[3], buff=0.3).to_edge(UP, buff=4.7),
		]

		# Creating NON-Normalized Scene
		self.add(dataset_group)
		self.wait()
		self.play(Create(cost_label.scale(1.15).next_to(r_graph, DOWN)))

		self.wait()

		# First Row Calc
		self.play(Create(costr_labels[0].move_to(2.5*DOWN + RIGHT/10)))
		self.play(FadeIn(costr_labels[1].next_to(cost_label, DOWN)))
		self.wait()

		# Second Row Calc
		self.play(FadeIn(costr_labels[2].next_to(costr_labels[1], DOWN)))
		self.wait(2)

		# Third Row Calc
		self.play(FadeIn(costr_labels[3].next_to(costr_labels[2], DOWN)))
		self.wait(2)

		# Last Row Calc
		self.play(FadeIn(costr_radius[0].move_to(2.5*DOWN + 5.5*RIGHT)))
		self.wait(2)

		# Show Graph
		r_graphs = [
			r_graph.get_implicit_curve(ravines[0], color=RED),
			r_graph.get_implicit_curve(ravines[1], color=GREEN),
			r_graph.get_implicit_curve(ravines[2], color=BLUE),
		]

		self.play(Create(r_graph))
		self.wait()
		self.play(Create(r_graphs[0]))
		self.wait(2)

		# Second Graph
		self.play(ReplacementTransform(costr_radius[0], costr_radius[1].move_to(costr_radius[0])))
		self.play(Create(r_graphs[1]))
		self.wait(2)

		# Third Graph
		self.play(ReplacementTransform(costr_radius[1], costr_radius[2].move_to(costr_radius[1])))
		self.play(Create(r_graphs[2]))
		self.wait(2)

		# ##################### Transition ################
		self.play(FadeOut(r_graphs[2]), 
			FadeOut(r_graphs[1]),
			FadeOut(r_graphs[0]),
			FadeOut(r_graph))

		self.play(FadeOut(cost_label))

		self.play(FadeOut(costr_radius[2]),
			FadeOut(costr_labels[3]),
			FadeOut(costr_labels[2]),
			FadeOut(costr_labels[1]),
			FadeOut(costr_labels[0]))


		# Creating Normalized Scene
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
		self.play(Create(costn_labels[0].move_to(2.5*DOWN + RIGHT/10)))
		self.play(FadeIn(costn_labels[1].next_to(cost_label, DOWN)))
		self.wait()

		# Second Row Calc
		self.play(FadeIn(costn_labels[2].next_to(costn_labels[1], DOWN)))
		self.wait(2)

		# Third Row Calc
		self.play(FadeIn(costn_labels[3].next_to(costn_labels[2], DOWN)))
		self.wait()

		# Last Row Calc
		self.play(FadeIn(costn_radius[0].move_to(2.5*DOWN + 5.5*RIGHT)))
		self.wait()


		# Show Graph
		n_graphs = [
			n_graph.get_implicit_curve(normals[0], color=RED),
			n_graph.get_implicit_curve(normals[1], color=GREEN),
			n_graph.get_implicit_curve(normals[2], color=BLUE),
		]

		self.play(Create(n_graphs[0]))
		self.wait(2)

		# Second Graph
		self.play(ReplacementTransform(costn_radius[0], costn_radius[1].move_to(costn_radius[0])))
		self.play(Create(n_graphs[1]))
		self.wait(2)

		# Third Graph
		self.play(ReplacementTransform(costn_radius[1], costn_radius[2].move_to(costn_radius[1])))
		self.play(Create(n_graphs[2]))
		self.wait(2)
