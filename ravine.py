from manim import *
import numpy as np


"""
A 3D function to produce a ravine graph

h/(1 + d*e^(xy)) + |l*x*y|, x*y <= 0
h/(1 + d*e^(-xy)) + |l*x*y|, x*y > 0
"""
def ravine(x, y):
	h = 5    # The height of the ravine, the bigger the higher
	d = 10   # The depth of the ravine, the bigger the deeper
	l = 0.1  # The steepness of the ravine, the bigger the steeper
	c = np.pow(-1, 1 - (x*y <= 0))

	return h/(1 + d*np.exp(c*x*y)) + np.abs(l*x*y)


"""
A 3D function to produce a regular convex parabolic graph

c*x^2 + c*y^2
"""
def convex(x, y):
	c = 0.1  # The width of the opening of the parabola, the smaller the wider
	return c*(np.power(x, 2) + np.power(y, 2))


class BraceAnnotation(Scene):
    def construct(self):
        dot = Dot([-2, -1, 0])
        dot2 = Dot([2, 1, 0])
        line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)
        b1 = Brace(line)
        b1text = b1.get_text("Horizontal distance")
        b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
        b2text = b2.get_tex("x-x_1")
        self.add(line, dot, dot2, b1, b2, b1text, b2text)
