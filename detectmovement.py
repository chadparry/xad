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
	ctx.clear(0., 0., 0.)

	horizontal_resolution, vertical_resolution = (60, 120)
	voxels = get_piece_voxels(horizontal_resolution, vertical_resolution)
	#orthographic = (voxels * 0.5).sum(axis=1)
	#cv2.imshow(WINNAME, orthographic)
	#key = cv2.waitKey(0)

	voxels_size = tuple(reversed(voxels.shape))
	texture = ctx.texture3d(voxels_size, 1, voxels, floats=True)
	texture.repeat_x = False
	texture.repeat_y = False
	texture.repeat_z = False
	texture.use()

	prog = ctx.program([
		ctx.vertex_shader('''
			#version 330

			in vec2 canvas_coord;
			out vec2 tex_coord;

			void main() {
				gl_Position = vec4(canvas_coord * 2. - 1., 0., 1.);
				tex_coord = canvas_coord;
			}
		'''),
		ctx.fragment_shader('''
			#version 330

			struct OptionalFloat {
				bool isPresent;
				float value;
			};

			uniform mat4 pose;
			uniform sampler3D voxels;

			in vec2 tex_coord;
			out vec4 color;

			void rotate(vec3 world_coord, mat4 rotation, out vec3 camera_coord) {
				vec4 hom_world_coord = vec4(world_coord, 1.);
				vec4 hom_camera_coord = rotation * hom_world_coord;
				camera_coord = hom_camera_coord.xyz / hom_camera_coord.w;
			}

			void interceptBoundingBoxFace(float intercept,
				float tail, float dir,
				vec2 other_tail, vec2 other_dir,
				out OptionalFloat dist
			) {
				if (dir == 0.) {
					dist.isPresent = false;
				} else {
					dist.value = (intercept - tail) / dir;
					vec2 other_intercept = other_tail + other_dir * dist.value;
					dist.isPresent = all(greaterThanEqual(other_intercept, vec2(0.))) &&
						all(lessThanEqual(other_intercept, vec2(1.)));
				}
			}

			void intersectBoundingBox(vec2 coord, mat4 rotation, vec3 voxels_size,
				out float min_z, out float max_z, out float depth, out int resolution
			) {
				vec3 tail, head;
				rotate(vec3(coord, 0.), rotation, tail);
				rotate(vec3(coord, 1.), rotation, head);
				vec3 dir = head - tail;

				OptionalFloat face_dist[6];
				interceptBoundingBoxFace(0., tail.x, dir.x, tail.yz, dir.yz, face_dist[0]);
				interceptBoundingBoxFace(1., tail.x, dir.x, tail.yz, dir.yz, face_dist[1]);
				interceptBoundingBoxFace(0., tail.y, dir.y, tail.xz, dir.xz, face_dist[2]);
				interceptBoundingBoxFace(1., tail.y, dir.y, tail.xz, dir.xz, face_dist[3]);
				interceptBoundingBoxFace(0., tail.z, dir.z, tail.xy, dir.xy, face_dist[4]);
				interceptBoundingBoxFace(1., tail.z, dir.z, tail.xy, dir.xy, face_dist[5]);

				bool any_intercepted = false;
				for (int i = 0; i < face_dist.length(); ++i) {
					if (face_dist[i].isPresent) {
						if (!any_intercepted || face_dist[i].value < min_z) {
							min_z = face_dist[i].value;
						}
						if (!any_intercepted || face_dist[i].value > max_z) {
							max_z = face_dist[i].value;
						}
						any_intercepted = true;
					}
				}

				if (any_intercepted) {
					vec3 min_intercept = tail + dir * min_z;
					vec3 max_intercept = tail + dir * max_z;
					vec3 boundingBoxTraversal = max_intercept - min_intercept;
					depth = length(boundingBoxTraversal);
					vec3 textureTraversal = voxels_size * boundingBoxTraversal;
					float voxelsTraversed = length(textureTraversal);
					resolution = int(ceil(voxelsTraversed));
				} else {
					resolution = 0;
				}
			}

			void main() {
				float min_z, max_z, depth;
				int resolution;
				intersectBoundingBox(tex_coord, pose, textureSize(voxels, 0), min_z, max_z, depth, resolution);
				float perceived;
				if (resolution > 0) {
					float step_z = (max_z - min_z) / resolution;
					float sum = 0.;
					for (int i = 0; i < resolution; ++i) {
						float z = min_z + (i + 0.5) * step_z;
						vec3 rot_tex_coord;
						rotate(vec3(tex_coord, z), pose, rot_tex_coord);
						sum += texture(voxels, rot_tex_coord).x;
					}
					perceived = depth * sum / resolution;
				} else {
					perceived = 0.;
				}
				color = vec4(vec3(perceived), 1.);
			}
		'''),
	])

	pose = numpy.float32([
		[1., 0., 0., 0.],
		[0., math.cos(math.pi/2.), -math.sin(math.pi/2.), 0.],
		[0., math.sin(math.pi/2.), math.cos(math.pi/2.), 0.],
		[0.0, 0., 0., 1.],
	]).transpose()
	prog.uniforms['pose'].write(pose)

	frame_size = 512
	fbo = ctx.framebuffer(ctx.renderbuffer((frame_size, frame_size), components=1))
	fbo.use()

	triangle_slice_vertices = numpy.float32([(0., 0.), (0., 1.), (1., 0.), (1., 1.)])
	vbo = ctx.buffer(triangle_slice_vertices)
	vao = ctx.simple_vertex_array(prog, vbo, ['canvas_coord'])
	vao.render(ModernGL.TRIANGLE_STRIP)

	data = fbo.read(components=1, floats=True)
	flipped_projection = numpy.frombuffer(data, dtype='float32').reshape(fbo.size)
	projection = numpy.flipud(numpy.fliplr(flipped_projection))
	cv2.imshow(WINNAME, projection * 4.)
	key = cv2.waitKey(0)


if __name__ == "__main__":
	main()
