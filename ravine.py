from manim import *
import numpy as np



class Ravine(ThreeDScene):
	"""
	A 3D function to produce a ravine graph

	h, d*x^2 + r*y^2 > w
	d*(l/w)*x^2 + r*(l/w)*y^2, d*x^2 + r*y^2 <= w
	"""
	def func(self, x, y, x_co=1, y_co=1, height=2, radius=4):
		if x_co*np.power(x, 2) + y_co*np.power(y, 2) > radius:
			return np.array([x, y, height])

		return np.array([x, y, (height/radius)*(x_co*np.power(x, 2) + y_co*np.power(y, 2))])



	def construct(self):
		axes = ThreeDAxes()
		convex_surface = ParametricSurface(
			lambda x, y: self.func(x, y),
			u_min=-3,
			u_max=3,
			v_min=-3,
			v_max=3,
			fill_opacity=0.5,
			checkerboard_colors=[BLUE, BLUE]
		)
		ravine_surface = ParametricSurface(
			lambda x, y: self.func(x, y, y_co=10, radius=10),
			u_min=-3,
			u_max=3,
			v_min=-3,
			v_max=3,
			fill_opacity=0.5,
			checkerboard_colors=[BLUE, BLUE]
		)

		x, y = 0.8, -0.8

		self.set_camera_orientation(phi=np.pi/5, theta=np.pi/5, distance=5)

		self.play(Create(axes))
		self.play(Create(convex_surface))

		d = Dot3D(radius=0.05, color=ORANGE).move_to(axes.coords_to_point(x, y, self.func(x, y)[-1] + 0.25))
		self.play(Create(d))

		self.begin_ambient_camera_rotation()
		self.wait(4)

		x, y = 2, -0.5
		d2 = Dot3D(radius=0.05, color=ORANGE).move_to(axes.coords_to_point(x, y, self.func(x, y, y_co=10, radius=10)[-1] + 0.5))
		self.play(Transform(convex_surface, ravine_surface))
		self.play(Transform(d, d2))

		self.wait()
		#self.move_camera(phi=np.pi/5, theta=0.45*np.pi)
		self.begin_ambient_camera_rotation()
		self.wait(5)
		#self.add(surface, axes, d)



