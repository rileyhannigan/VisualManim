from manim import *

class Ravines(Scene):	
	def construct(self):

		col1_vals = [114.8, 0.00323, 5.1]
		col2_vals = [338.1, 0.00183, 3.2]
		col3_vals = [98.8, 0.00279, 4.1]
		dataset = DecimalTable(
			[col1_vals, col2_vals, col3_vals],
			col_labels = [MathTex("x_1"), MathTex("x_2"), MathTex("t")],
			h_buff = 1,
			element_to_mobject_config={"num_decimal_places": 5})


		w1_label=MathTex("w_1 \\leftarrow", "w_1 - \\alpha \\frac{1}{2} (x_1(w_1 x_1 + w_2 x_2 - t)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label1=MathTex("w_1 \\leftarrow", "w_1 - \\alpha \\frac{1}{2} (x_1(w_1 x_1 + w_2 x_2 - t)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label2=MathTex("w_1 \\leftarrow", "0.02 - 0.0005 (x_1(0.02 x_1 + 0.1 x_2 - t)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label3=MathTex("w_1 \\leftarrow", "0.02 - 0.0005 (114.8(0.02 * 114.8 + 0.1 * 0.00323 - 5.1)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label4=MathTex("w_1 \\leftarrow", "0.18095").next_to(dataset,DOWN, buff=-0.5)

		w2_label=MathTex("w_2 \\leftarrow", "w_2 - \\alpha \\frac{1}{2} (x_2(w_1 x_1 + w_2 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label1=MathTex("w_2 \\leftarrow", "w_2 - \\alpha \\frac{1}{2} (x_2(w_1 x_1 + w_2 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label2=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (x_2(0.01 x_1 + 0.01 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label3=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (0.00323(0.01 * 114.8 + 0.01 * 0.00323 - 5.1)) ").next_to(w1_label,DOWN)
		w2_label4=MathTex("w_2 \\leftarrow", "0.010004 ").next_to(w1_label,DOWN)

		y_label=MathTex("y =", "w_1 x_1 + w_2 x_2").next_to(w2_label, DOWN)
		

		misc_label=MathTex("w_1 = 0.02, w_2 = 0.01, \\alpha = 0.001")
		

		graph = Axes(
			x_range = [0, 10, 2], 
			y_range = [0, 10, 2],
			x_length = 10,
			y_length = 10,
			axis_config={"include_numbers": False})
		labels = graph.get_axis_labels(x_label='w_1', y_label='w_2')


		ellipse_1 = Ellipse(width=2.0, height=8.0, color=BLUE_A)
		ellipse_2 = Ellipse(width=1.5, height=7.0, color=BLUE_C)
		ellipse_3 = Ellipse(width=1.0, height=6.0, color=BLUE_D)
		ellipse_4 = Ellipse(width=0.5, height=5.0, color=BLUE_E)



		graph_group = Group(graph,labels, ellipse_1, ellipse_2, ellipse_3, ellipse_4)
		dataset_group = Group(dataset, w1_label, w2_label)
		group1 = Group(dataset_group, graph_group).scale(0.6).arrange(buff=1).to_edge(UP, buff=0.5)
		

		self.add(group1)
		self.wait()
		self.play(Transform(w1_label, w1_label1))
		self.play(Transform(w2_label, w2_label1))
		self.wait()
		self.play(Transform(w1_label, w1_label2))
		self.play(Transform(w2_label, w2_label2))
		self.wait()
		self.play(Transform(w1_label, w1_label3))
		self.play(Transform(w2_label, w2_label3))
		self.wait()
		self.play(Transform(w1_label, w1_label4))
		self.play(Transform(w2_label, w2_label4))
		


