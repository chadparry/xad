#!/usr/bin/env python3

import cv2
import ModernGL
import numpy
import scipy.stats

import heatmaps


EDGE_RESOLUTION = 10
EDGE_GAUSSIAN_SCALE = 4


def get_light_square_heatmap(frame_size, projection):
	return get_square_heatmap(frame_size, projection, False)


def get_dark_square_heatmap(frame_size, projection):
	return get_square_heatmap(frame_size, projection, True)


def get_square_heatmap(frame_size, projection, is_offset):
	ctx = ModernGL.create_standalone_context()

	weights = numpy.fromiter(
		(scipy.stats.norm.cdf((weight_idx / EDGE_RESOLUTION) * EDGE_GAUSSIAN_SCALE)
			for weight_idx in range(EDGE_RESOLUTION - 1, -1, -1)),
		dtype=numpy.float32).reshape((1, EDGE_RESOLUTION))
	weights_size = tuple(reversed(weights.shape))
	texture = ctx.texture(weights_size, 1, weights, floats=True)
	texture.repeat_x = False
	texture.use()

	prog = ctx.program([
		ctx.vertex_shader('''
			#version 330

			uniform mat4 projection;

			in vec2 board_coord;
			out vec2 tex_coord;

			void main() {
				vec4 homog_coord = vec4(board_coord, 0, 1);
				gl_Position = projection * homog_coord;
				tex_coord = board_coord;
			}
		'''),
		ctx.fragment_shader('''
			#version 330

			uniform sampler2D weights;

			in vec2 tex_coord;
			out float color;

			void get_axis_weight(float location, sampler2D axis_weights, out float weight) {
				float square_location = mod(location, 1);
				float tex_index = abs(0.5 - square_location) * 2;
				weight = texture(axis_weights, vec2(tex_index, 0)).x;
			}

			void main() {
				float hor_weight, ver_weight;
				get_axis_weight(tex_coord.x, weights, hor_weight);
				get_axis_weight(tex_coord.y, weights, ver_weight);
				color = hor_weight * ver_weight;
			}
		'''),
	])

	fbo = ctx.framebuffer(ctx.renderbuffer(frame_size, components=1))
	fbo.use()

	squares = ((x, y)
		for y in range(8)
		for x in range(8)
		if bool((x + y) % 2) != is_offset)
	triangles = numpy.float32([
		[
			[
				(square[0], square[1]),
				(square[0], square[1] + 1),
				(square[0] + 1, square[1] + 1),
			],
			[
				(square[0] + 1, square[1] + 1),
				(square[0] + 1, square[1]),
				(square[0], square[1]),
			],
		]
		for square in squares
	])
	vbo = ctx.buffer(triangles)
	vao = ctx.simple_vertex_array(prog, vbo, ['board_coord'])

	(rotation, jacobian) = cv2.Rodrigues(projection.pose.rvec)
	projection_matrix = numpy.dot(
			projection.cameraIntrinsics.cameraMatrix,
			numpy.hstack([rotation, projection.pose.tvec]),
		).astype(numpy.float32)
	# Change the frame coordinates from (0, 0) - frame_size to (-1, -1) - (1, 1)
	gl_shift = numpy.float32([
		[1, 0, -1],
		[0, 1, -1],
		[0, 0, 1],
	])
	gl_scale = numpy.float32([
		[2 / frame_size[0], 0, 0],
		[0, 2 / frame_size[1], 0],
		[0, 0, 1],
	])
	scaled_matrix = numpy.dot(numpy.dot(gl_shift, gl_scale), projection_matrix)
	gl_projection = numpy.vstack([
		scaled_matrix[:2],
		numpy.zeros(4, dtype=numpy.float32),
		scaled_matrix[2:],
	])
	prog.uniforms['projection'].write(gl_projection.transpose())

	ctx.clear()
	vao.render()

	data = fbo.read(components=1, floats=True)
	projection_shape = tuple(reversed(frame_size))
	heatmap = numpy.frombuffer(data, dtype=numpy.float32).reshape(projection_shape)
	return heatmaps.Heatmap(heatmap).as_sparse()
