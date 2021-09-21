from manim import *
import numpy as np


"""
h/(1 + d*e^(-xy)) + |l*x*y|
"""
def ravine(x, y):
	h = 5
	d = 10
	l = 0.1

	z = np.abs(l*x*y)

	if x*y <= 0:
		z += h/(1 + d*np.exp(x*y))
	else:
		z += h/(1 + d*np.exp(-1*x*y))

	return z


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
