#!/usr/bin/env python3

import chess
import cv2
import functools
import itertools
import math
import ModernGL
import numpy
import operator
import scipy.stats

import pose
import size


WINNAME = 'Chess Transcription'
GAUSSIAN_SCALE = 4.


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
	getters = [memoized_radial_attenuation, taper_attenuation, height_attenuation, reflective_attenuation]
	factors = (getter(idx, *coords) for getter in getters)
	weight = functools.reduce(operator.mul, factors, 1.)
	return weight


def get_horizontal_coord(h, horizontal_resolution):
	return (h + 1) / (horizontal_resolution + 1)


def get_vertical_coord(v, vertical_resolution):
	return (v - vertical_resolution) / (vertical_resolution + 1)


def get_coords(idx, horizontal_resolution, vertical_resolution):
	return (
		get_horizontal_coord(idx[0], horizontal_resolution),
		get_horizontal_coord(idx[1], horizontal_resolution),
		get_vertical_coord(idx[2], vertical_resolution),
	)


def get_memoized_radial_attenuation(horizontal_resolution):
	idxs = ((x, y) for y in range((horizontal_resolution + 1) // 2) for x in range(y + 1))
	GAUSSION_DENOMINATOR = 1 - 2 * scipy.stats.norm.cdf(-numpy.linalg.norm((0.5, 0.5)) * GAUSSIAN_SCALE)
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


def taper_attenuation(idx, x, y, z):
	return 1 - abs(z) * math.sqrt((0.5 - x)**2 + (0.5 - y)**2)


def height_attenuation(idx, x, y, z):
	MIN_RELATIVE_HEIGHT = 9 / 11
	return 1 - max(abs(z) - MIN_RELATIVE_HEIGHT, 0) / (1 - MIN_RELATIVE_HEIGHT)


def reflective_attenuation(idx, x, y, z):
	return 0.3 if z < 0 else 1.


def get_piece_heatmaps(frame_size, projection):
	rotation, jacobian = cv2.Rodrigues(projection.pose.rvec)
	inv_rotation = rotation.transpose()
	inv_tvec = numpy.dot(inv_rotation, -projection.pose.tvec)
	inv_pose = numpy.vstack([numpy.hstack([inv_rotation, inv_tvec]), numpy.float32([0, 0, 0, 1])])
	inv_camera_matrix = numpy.linalg.inv(projection.cameraIntrinsics.cameraMatrix)
	# Change the frame coordinates from (0, 0) - frame_size to (-1, -1) - (1, 1)
	gl_scale = numpy.float32([
		[frame_size[0] / 2, 0, 0],
		[0, frame_size[1] / 2, 0],
		[0, 0, 1],
	])
	gl_shift = numpy.float32([
		[1, 0, 1],
		[0, 1, 1],
		[0, 0, 1],
	])
	gl_inv_camera_matrix = numpy.dot(inv_camera_matrix, numpy.dot(gl_scale, gl_shift))
	ext_inv_camera_matrix = numpy.vstack([gl_inv_camera_matrix, numpy.float32([0, 0, 1])])

	horizontal_resolution, vertical_resolution = (12, 24)
	voxels = get_piece_voxels(horizontal_resolution, vertical_resolution)

	# This helper performs volume ray casting
	ctx = ModernGL.create_standalone_context()

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
				gl_Position = vec4(canvas_coord, 0., 1.);
				tex_coord = canvas_coord;
			}
		'''),
		ctx.fragment_shader('''
			#version 330

			struct OptionalFloat {
				bool isPresent;
				float value;
			};

			uniform mat3 inv_projection;
			uniform vec3 camera_position;
			uniform vec3 piece_dimensions;
			uniform sampler3D voxels;

			in vec2 tex_coord;
			out float color;

			void reverse_project(vec2 image_coord, mat3 rotation, vec3 camera_position, out vec3 camera_ray) {
				vec3 hom_image_coord = vec3(image_coord, 1);
				vec3 world_coord = rotation * hom_image_coord;
				camera_ray = world_coord - camera_position;
			}

			void interceptBoundingBoxFace(float intercept,
				float tail, float dir,
				vec2 other_tail, vec2 other_dir,
				out OptionalFloat dist
			) {
				if (dir == 0) {
					dist.isPresent = false;
				} else {
					dist.value = (intercept - tail) / dir;
					vec2 other_intercept = other_tail + other_dir * dist.value;
					dist.isPresent = all(greaterThanEqual(other_intercept, vec2(0))) &&
						all(lessThanEqual(other_intercept, vec2(1)));
				}
			}

			void intersectBoundingBox(vec3 tail, vec3 dir, out float min_z, out float max_z) {
				OptionalFloat face_dist[6];
				interceptBoundingBoxFace(0, tail.x, dir.x, tail.yz, dir.yz, face_dist[0]);
				interceptBoundingBoxFace(1, tail.x, dir.x, tail.yz, dir.yz, face_dist[1]);
				interceptBoundingBoxFace(0, tail.y, dir.y, tail.xz, dir.xz, face_dist[2]);
				interceptBoundingBoxFace(1, tail.y, dir.y, tail.xz, dir.xz, face_dist[3]);
				interceptBoundingBoxFace(0, tail.z, dir.z, tail.xy, dir.xy, face_dist[4]);
				interceptBoundingBoxFace(1, tail.z, dir.z, tail.xy, dir.xy, face_dist[5]);

				bool none_intercepted = true;
				for (int i = 0; i < face_dist.length(); ++i) {
					if (face_dist[i].isPresent) {
						if (none_intercepted || face_dist[i].value < min_z) {
							min_z = face_dist[i].value;
						}
						if (none_intercepted || face_dist[i].value > max_z) {
							max_z = face_dist[i].value;
						}
						none_intercepted = false;
					}
				}
				if (none_intercepted || max_z < 0) {
					discard;
				}
				min_z = max(min_z, 0);
			}

			void getBoundingBoxTraversal(vec3 tail, vec3 dir, float min_z, float max_z, out vec3 boundingBoxTraversal) {
				vec3 min_intercept = tail + dir * min_z;
				vec3 max_intercept = tail + dir * max_z;
				boundingBoxTraversal = max_intercept - min_intercept;
			}

			void getResolution(vec3 boundingBoxTraversal, vec3 textureSize, out int resolution) {
				vec3 textureTraversal = textureSize * boundingBoxTraversal;
				float voxelsTraversed = length(textureTraversal);
				resolution = int(ceil(voxelsTraversed));
			}

			void main() {
				vec3 camera_ray;
				reverse_project(tex_coord, inv_projection, camera_position, camera_ray);

				float min_z, max_z;
				intersectBoundingBox(camera_position, camera_ray, min_z, max_z);

				vec3 boundingBoxTraversal;
				getBoundingBoxTraversal(camera_position, camera_ray, min_z, max_z, boundingBoxTraversal);

				int resolution;
				getResolution(boundingBoxTraversal, textureSize(voxels, 0), resolution);
				float depth = length(boundingBoxTraversal * piece_dimensions);
				float exp = depth / resolution;

				float step_z = (max_z - min_z) / resolution;
				float transparency = 1;
				for (int i = 0; i < resolution; ++i) {
					float z = min_z + (i + 0.5) * step_z;
					vec3 world_coord = camera_position + z * camera_ray;
					float voxel = texture(voxels, world_coord).x;
					transparency *= pow(1 - voxel, exp);
				}

				color = 1 - transparency;
			}
		'''),
	])

	fbo = ctx.framebuffer(ctx.renderbuffer(frame_size, components=1))
	fbo.use()

	triangle_slice_vertices = numpy.float32([(-1, -1), (-1, 1), (1, -1), (1, 1)])
	vbo = ctx.buffer(triangle_slice_vertices)
	vao = ctx.simple_vertex_array(prog, vbo, ['canvas_coord'])

	projection_shape = tuple(reversed(frame_size))

	heatmaps = {}
	for (square, piece_type) in itertools.product(chess.SQUARES, chess.PIECE_TYPES):
		i = chess.square_file(square)
		j = chess.square_rank(square)
		height = size.HEIGHTS[piece_type]

		# Transform the piece to a cube at the origin, to simplify the ray caster
		RELATIVE_REFLECTION_HEIGHT = 0.5
		piece_scale = height / (1 - RELATIVE_REFLECTION_HEIGHT)
		shift = numpy.float32([
			[1, 0, 0, -i],
			[0, 1, 0, -j],
			[0, 0, 1, RELATIVE_REFLECTION_HEIGHT],
			[0, 0, 0, 1],
		])
		stretch = numpy.float32([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1 / piece_scale, 0],
			[0, 0, 0, 1],
		])
		piece_inv_pose = numpy.dot(numpy.dot(shift, stretch), inv_pose)
		piece_inv_projection = numpy.dot(piece_inv_pose, ext_inv_camera_matrix)
		camera_position = numpy.dot(piece_inv_pose, numpy.float32([0, 0, 0, 1]).reshape(4,1))

		prog.uniforms['inv_projection'].write(piece_inv_projection[:3].transpose())
		prog.uniforms['camera_position'].write(camera_position[:3])
		prog.uniforms['piece_dimensions'].write(numpy.float32([1, 1, piece_scale]))

		ctx.clear()
		# TODO: It would be possible to speed this up by shrinking the frame
		# until it barely contains the bounding box
		vao.render(ModernGL.TRIANGLE_STRIP)

		data = fbo.read(components=1, floats=True)
		heatmap = numpy.frombuffer(data, dtype='float32').reshape(projection_shape)

		heatmaps[(square, piece_type)] = heatmap

	return heatmaps


def get_depths(projection):
	rotation, jacobian = cv2.Rodrigues(projection.pose.rvec)
	inv_rotation = rotation.transpose()
	camera_position = numpy.dot(inv_rotation, -projection.pose.tvec)
	(camera_x, camera_y) = camera_position[:2]

	distances = []
	for square in chess.SQUARES:
		i = chess.square_file(square)
		j = chess.square_rank(square)
		distance = math.sqrt((camera_x - i)**2 + (camera_y - j)**2)
		distances.append((distance, square))

	order = sorted(distances)
	depths = {square: idx for (idx, (distance, square)) in enumerate(order)}
	return depths


def normalize_diff(diff):
	"""The diff will be normalized so the average is zero and the standard deviation is one"""
	diff_sum = diff.sum()
	if diff_sum:
		centered_diff = diff - diff_sum / diff.size
		standard_deviation = math.sqrt((centered_diff**2).sum() / diff.size)
		normalized_diff = centered_diff / standard_deviation
	else:
		normalized_diff = diff
	return normalized_diff


def get_negative_composite(negative_composite_memo, heatmaps, sorted_pieces):
	if sorted_pieces in negative_composite_memo:
		#print('        CACHE HIT ', sorted_pieces)
		return negative_composite_memo[sorted_pieces]
	#print('        CACHE MISS', sorted_pieces)

	(sorted_pieces_tail, sorted_pieces_head) = (sorted_pieces[:-1], sorted_pieces[-1])
	negative_composite_tail = get_negative_composite(negative_composite_memo, heatmaps, sorted_pieces_tail)
	negative_composite_head = heatmaps[sorted_pieces_head]
	negative_composite = negative_composite_tail * (1 - negative_composite_head)
	negative_composite_memo[sorted_pieces] = negative_composite
	return negative_composite


def get_piece_diff(negative_composite_memo, heatmaps, depths, sorted_pieces, focal_piece):
	depth = depths[focal_piece[0]]
	occluding_pieces = tuple(itertools.takewhile(
		lambda piece_item: depths[piece_item[0]] < depth,
		sorted_pieces,
	))
	negative_composite = get_negative_composite(negative_composite_memo, heatmaps, occluding_pieces)
	negative_layer = 1 - heatmaps[focal_piece]
	combined_negative_composite = negative_composite * negative_layer
	diff = 1 - (negative_composite - combined_negative_composite)
	return diff


def get_move_diffs(heatmaps, depths, negative_composite_memo, board):
	move_diffs = {}
	sorted_pieces = sorted((
		(piece_item[0], piece_item[1].piece_type)
			for piece_item in board.piece_map().items()),
		key=lambda piece_item: depths[piece_item[0]])
	for move in board.legal_moves:
		pieces_before = board.piece_map().items()
		board.push(move)
		try:
			pieces_after = board.piece_map().items()
		finally:
			board.pop()
		moved_piece_items = pieces_before ^ pieces_after
		moved_pieces = [(square, piece.piece_type) for (square, piece) in moved_piece_items]
		stable_pieces = [piece for piece in sorted_pieces if piece not in moved_pieces]
		piece_diffs = [get_piece_diff(negative_composite_memo, heatmaps, depths, stable_pieces, piece_item)
			for piece_item in moved_pieces]

		combined_diff = 1 - numpy.prod(piece_diffs, axis=0)
		normalized_diff = normalize_diff(combined_diff)
		move_diffs[move] = normalized_diff

	return move_diffs


def main():
	frame_size = (1280, 720)

	# Projection found by findboard
	projection = pose.Projection(
		pose.CameraIntrinsics(
			cameraMatrix=numpy.float32([
				[887.09763773,   0.        , 639.5       ],
				[  0.        , 887.09763773, 359.5       ],
				[  0.        ,   0.        ,   1.        ],
			]),
			distCoeffs=numpy.float32([0., 0., 0., 0., 0.]),
		),
		pose.Pose(
			rvec=numpy.float32([
				[ 1.32300998],
				[-1.32785091],
				[ 1.14510022],
			]),
			tvec=numpy.float32([
				[ 3.58316198],
				[ 3.06215196],
				[10.00036672],
			])
		),
	)

	subtractor = cv2.imread('diff.png')[:,:,0]
	normalized_subtractor = normalize_diff(subtractor)

	heatmaps = get_piece_heatmaps(frame_size, projection)
	depths = get_depths(projection)
	board = chess.Board()
	projection_shape = tuple(reversed(frame_size))
	negative_composite_memo = {(): numpy.ones(projection_shape)}
	move_diffs = get_move_diffs(heatmaps, depths, negative_composite_memo, board)
	for (move, move_diff) in move_diffs.items():
		# The Pearson correlation coefficient measures the goodness of fit
		score = (move_diff * normalized_subtractor).mean()
		print('score', board.san(move), score)
		composite = numpy.zeros((720, 1280, 3), dtype='float32')
		composite[:,:,2] += move_diff
		composite[:,:,1] += normalized_subtractor
		cv2.imshow(WINNAME, composite / 20)
		key = cv2.waitKey(0)


if __name__ == "__main__":
	main()
