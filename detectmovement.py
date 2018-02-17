#!/usr/bin/env python3

import cv2
import functools
import itertools
import math
import ModernGL
import numpy
import operator
import scipy.stats


WINNAME = 'Chess Transcription'
GAUSSIAN_SCALE = 3.


def get_piece_voxels(horizontal_resolution, vertical_resolution):
	"""Plots a piece in a 3D matrix according to the likelihood that it appears in that location"""
	# The vertical shape is doubled to make room for a reflection
	shape = (vertical_resolution * 2 + 1, horizontal_resolution, horizontal_resolution)
	memoized_radial_attenuation = get_memoized_radial_attenuation(horizontal_resolution)
	reversed_idxs = itertools.product(*(range(dim) for dim in shape))
	# Indices are reversed from (z, y, x) like in the shape to (x, y, z)
	idxs = (list(reversed(idx)) for idx in reversed_idxs)
	weights = (get_voxel_weight(memoized_radial_attenuation, idx, get_coords(idx, horizontal_resolution, vertical_resolution))
		for idx in idxs)
	voxel_count = numpy.prod(shape)
	voxels = numpy.fromiter(weights, dtype='float32', count=voxel_count).reshape(shape)
	return voxels


def get_voxel_weight(memoized_radial_attenuation, idx, coords):
	getters = [memoized_radial_attenuation, height_attenuation, reflective_attenuation]
	factors = (getter(idx, *coords) for getter in getters)
	weight = functools.reduce(operator.mul, factors, 1.)
	return weight


def get_horizontal_coord(h, horizontal_resolution):
	return (h + 1) / float(horizontal_resolution + 1)


def get_vertical_coord(v, vertical_resolution):
	return (v - vertical_resolution) / float(vertical_resolution + 1)


def get_coords(idx, horizontal_resolution, vertical_resolution):
	return (
		get_horizontal_coord(idx[0], horizontal_resolution),
		get_horizontal_coord(idx[1], horizontal_resolution),
		get_vertical_coord(idx[2], vertical_resolution),
	)


def get_memoized_radial_attenuation(horizontal_resolution):
	idxs = ((x, y) for y in range((horizontal_resolution + 1) // 2) for x in range(y + 1))
	GAUSSION_DENOMINATOR = 1. - 2. * scipy.stats.norm.cdf(-numpy.linalg.norm((0.5, 0.5)) * GAUSSIAN_SCALE)
	def get_clamped_gaussian(h):
		return scipy.stats.norm.pdf(h * GAUSSIAN_SCALE) / GAUSSION_DENOMINATOR
	memo = {(x, y):
		get_clamped_gaussian(numpy.linalg.norm([
			0.5 - get_horizontal_coord(h, horizontal_resolution)
			for h in (x, y)
		]))
		for (x, y) in idxs}
	def memoized_radial_attenuation(idx, x, y, z):
		# Take advantage of the 8-fold symmetry of the radial distance
		(mirrored_x, mirrored_y) = (
			(horizontal_resolution - 1 - abs(horizontal_resolution - 1 - h * 2)) // 2
			for h in idx[:2])
		result = memo[(min(mirrored_x, mirrored_y), max(mirrored_x, mirrored_y))]
		return result
	return memoized_radial_attenuation


def height_attenuation(idx, x, y, z):
	return 1. - abs(z)


def reflective_attenuation(idx, x, y, z):
	return 0.3 if z <= 0 else 1.


def main():
	ctx = ModernGL.create_standalone_context()

	horizontal_resolution, vertical_resolution = (60, 120)
	voxels = get_piece_voxels(horizontal_resolution, vertical_resolution)
	#orthographic = (voxels * 0.5).sum(axis=1)
	#cv2.imshow(WINNAME, orthographic)
	#key = cv2.waitKey(0)

	voxels_size = tuple(reversed(voxels.shape))
	texture = ctx.texture3d(voxels_size, 1, voxels / 10., floats=True)
	texture.repeat_x = False
	texture.repeat_y = False
	texture.use()

	prog = ctx.program([
		ctx.vertex_shader('''
			#version 330

			in vec3 canvas_coord;
			in vec3 tex_coord;
			out vec3 v_tex_coord;

			void main() {
				gl_Position = vec4(canvas_coord, 1.);
				v_tex_coord = tex_coord;
			}
		'''),
		ctx.fragment_shader('''
			#version 330

			uniform mat4 pose;
			uniform sampler3D voxels;

			in vec3 v_tex_coord;
			out vec4 color;

			void main() {
				vec4 hom_rot_tex_coord;
				hom_rot_tex_coord = pose * vec4(v_tex_coord, 1.);
				vec3 rot_tex_coord;
				rot_tex_coord = hom_rot_tex_coord.xyz / hom_rot_tex_coord.w;
				float alpha;
				if (any(lessThan(rot_tex_coord, vec3(0., 0., 0.))) ||
					any(greaterThan(rot_tex_coord, vec3(1., 1., 1.)))) {
					alpha = 0.;
				} else {
					alpha = texture(voxels, rot_tex_coord).x;
				}
				color = vec4(0., 0., 0., alpha);
			}
		'''),
	])

	prog.uniforms['pose'].write(numpy.float32([
		[2., 0., 0., 0.],
		[0., math.sin(math.pi/6.), -math.cos(math.pi/6.), 0.],
		[0., math.cos(math.pi/6.), math.sin(math.pi/6.), 0.],
		[0., 0., 0., 1.],
	]).transpose())

	canvas_anchor = (-1., -1.)
	sw_triangle = [(1., 0.), (0., 0.), (0., 1.)]
	ne_triangle = [(0., 1.), (1., 1.), (1., 0)]
	triangle_vertices = list(itertools.chain(sw_triangle, ne_triangle))
	slice_depths = ((get_vertical_coord(z, vertical_resolution), z / float(voxels.shape[0] - 1))
		for z in range(voxels.shape[0]))
	slices_iter = ((
			canvas_anchor[0] + texture_xy[0] * 2.,
			canvas_anchor[1] + texture_xy[1] * 2.,
			canvas_z,
			texture_xy[0],
			texture_xy[1],
			texture_z,
		)
		for (canvas_z, texture_z) in slice_depths
		for texture_xy in triangle_vertices)
	slices_flattened_iter = (v for coord in slices_iter for v in coord)
	slices_value_count = voxels.shape[0] * len(triangle_vertices) * 6
	slices = numpy.fromiter(slices_flattened_iter, dtype='float32', count=slices_value_count)

	vbo = ctx.buffer(slices)
	vao = ctx.simple_vertex_array(prog, vbo, ['canvas_coord', 'tex_coord'])

	frame_size = 512
	fbo = ctx.framebuffer(ctx.renderbuffer((frame_size, frame_size)))

	fbo.use()
	ctx.enable(ModernGL.BLEND)
	ctx.clear(1., 1., 1.)
	vao.render()

	data = fbo.read(components=3, floats=True)
	output_shape = (frame_size, frame_size, 3)
	flipped_rgb_projection = numpy.frombuffer(data, dtype='float32').reshape(output_shape)
	flipped_projection = flipped_rgb_projection[:,:,0]
	projection = numpy.flipud(numpy.fliplr(flipped_projection))
	cv2.imshow(WINNAME, projection)
	key = cv2.waitKey(0)


if __name__ == "__main__":
	main()
