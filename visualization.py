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

		highlight = SurroundingRectangle(dataset[1], buff = .1)


		w1_label=MathTex("w_1 \\leftarrow", "w_1 - \\alpha \\frac{1}{2} (x_1(w_1 x_1 + w_2 x_2 - t)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label1=MathTex("{{w_1 \\leftarrow}} {{w_1}} - {{\\alpha}} \\frac{1}{2} (x_1({{w_1}} x_1 + {{w_2}} x_2 - t)) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label1.set_color_by_tex("w_1", YELLOW)
		w1_label1.set_color_by_tex("w_2", YELLOW)
		w1_label1.set_color_by_tex("\\alpha", YELLOW)

		w1_label2=MathTex("w_1 \\leftarrow", "{{0.02}} - {{0.0005}} ({{x_1}}({{0.02}} {{x_1}} + {{0.01}} {{x_2}} - {{t}})) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label2.set_color_by_tex("0.02", YELLOW)
		w1_label2.set_color_by_tex("0.0005", YELLOW)
		w1_label2.set_color_by_tex("0.01", YELLOW)
		w1_label2.set_color_by_tex("x_1", RED)
		w1_label2.set_color_by_tex("x_2", RED)
		w1_label2.set_color_by_tex("t", RED)

		w1_label3=MathTex("w_1 \\leftarrow", "{{0.02}} - {{0.0005}} ({{114.8}}({{0.02}} * {{114.8}} + {{0.01}} * {{0.00323}} - {{5.1}})) ").next_to(dataset,DOWN, buff=-0.5)
		w1_label3.set_color_by_tex("0.02", YELLOW)
		w1_label3.set_color_by_tex("0.0005", YELLOW)
		w1_label3.set_color_by_tex("0.01", YELLOW)
		w1_label3.set_color_by_tex("114.8", RED)
		w1_label3.set_color_by_tex("0.00323", RED)
		w1_label3.set_color_by_tex("5.1", RED)

		w1_label4=MathTex("w_1 \\leftarrow", "0.18095").next_to(dataset,DOWN, buff=-0.5)

		w2_label=MathTex("w_2 \\leftarrow", "w_2 - \\alpha \\frac{1}{2} (x_2(w_1 x_1 + w_2 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label1=MathTex("w_2 \\leftarrow", "w_2 - \\alpha \\frac{1}{2} (x_2(w_1 x_1 + w_2 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label2=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (x_2(0.02 x_1 + 0.01 x_2 - t)) ").next_to(w1_label,DOWN)
		w2_label3=MathTex("w_2 \\leftarrow", "0.01 - 0.0005 (0.00323(0.02 * 114.8 + 0.01 * 0.00323 - 5.1)) ").next_to(w1_label,DOWN)
		w2_label4=MathTex("w_2 \\leftarrow", "0.010005 ").next_to(w1_label,DOWN)

		y_label=MathTex("y =", "w_1 x_1 + w_2 x_2").next_to(w2_label, DOWN)
		

		misc_label=MathTex("\\alpha = 0.001, w_1 = 0.02, w_2 = 0.01").next_to(dataset,UP)
		misc_label.set_color_by_tex("w_1", YELLOW)
		misc_label2=MathTex("\\alpha = 0.001, w_1 = 0.18095, w_2 = 0.010005").next_to(dataset,UP).scale(0.6).to_edge(LEFT, buff=1).to_edge(UP, buff=1)
		misc_label2.set_color_by_tex("w_1", YELLOW)

		

		graph = Axes(
			x_range = [0, 10, 2], 
			y_range = [0, 10, 2],
			x_length = 10,
			y_length = 10,
			axis_config={"include_numbers": False})
		labels = graph.get_axis_labels(x_label='w_1', y_label='w_2')
		graph_label = MathTex("y = w_1 x_1 + w_2 x_2").to_edge(RIGHT, buff=1).set_color_by_tex("y", BLUE_D)




		ellipse_1a = Ellipse(width=2.0, height=8.0, color=BLUE_A)
		ellipse_2a = Ellipse(width=1.5, height=7.0, color=BLUE_C)
		ellipse_3a = Ellipse(width=1.0, height=6.0, color=BLUE_D)
		ellipse_4a = Ellipse(width=0.5, height=5.0, color=BLUE_E)

		ellipse_1b = Ellipse(width=1.5, height=8.0, color=BLUE_A).scale(0.6).to_edge(RIGHT, buff=4).to_edge(UP, buff=1)
		ellipse_2b = Ellipse(width=1.0, height=7.0, color=BLUE_C).scale(0.6).to_edge(RIGHT, buff=4.166).to_edge(UP, buff=1.333)
		ellipse_3b = Ellipse(width=0.5, height=6.0, color=BLUE_D).scale(0.6).to_edge(RIGHT, buff=4.333).to_edge(UP, buff=1.666)
		ellipse_4b = Ellipse(width=0.25, height=5.0, color=BLUE_E).scale(0.6).to_edge(RIGHT, buff=4.4).to_edge(UP, buff=2)

		graph_arrow = CurvedArrow(start_point=np.array([2, -3, 0]), end_point=np.array([5, 0, 0]))
		dataset_arrow = CurvedArrow(start_point=np.array([0, -1.5, 0]), end_point=np.array([-1.5, 2.5, 0]))


		graph_group = Group(graph,labels, ellipse_1a, ellipse_2a, ellipse_3a, ellipse_4a, graph_label)
		dataset_group = Group(dataset, misc_label, w1_label, w2_label)
		group1 = Group(dataset_group, graph_group).scale(0.6).arrange(buff=1).to_edge(UP, buff=0.5)
		


		self.add(group1)
		self.wait()
		self.play(Transform(w1_label, w1_label1))
		self.play(Transform(w2_label, w2_label1))
		self.wait(2)
		self.play(Transform(w1_label, w1_label2))
		self.play(Transform(w2_label, w2_label2))
		dataset.add(SurroundingRectangle(dataset.get_rows()[1], color="RED"))
		self.wait(2)
		self.play(Transform(w1_label, w1_label3))
		self.play(Transform(w2_label, w2_label3))
		self.wait(2)
		self.play(Transform(w1_label, w1_label4))
		self.play(Transform(w2_label, w2_label4))
		self.play(Create(graph_arrow))
		self.play(Transform(ellipse_1a, ellipse_1b))
		self.play(Transform(ellipse_2a, ellipse_2b))
		self.play(Transform(ellipse_3a, ellipse_3b))
		self.play(Transform(ellipse_4a, ellipse_4b))
		self.play(Create(dataset_arrow))
		self.play(Transform(misc_label, misc_label2))
		


