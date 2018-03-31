#!/usr/bin/env python3

import chess
import collections
import cv2
import functools
import itertools
import math
import ModernGL
import numpy
import operator

import pose
import size
import voxels


WINNAME = 'Chess Transcription'


class Heatmap:
	"""Allows multiplication on a very sparse matrix, more efficient for this case than scipy.sparse"""

	def __init__(self, delegate, init_slice=None, shape=None):
		self.delegate = delegate
		self.shape = self.delegate.shape if shape is None else shape
		self.slice = numpy.s_[0:self.shape[0],0:self.shape[1]] if init_slice is None else init_slice

	def as_sparse(self):
		(y_coords, x_coords) = numpy.where(self.delegate)
		y_start = y_coords.min()
		y_stop = y_coords.max() + 1
		x_start = x_coords.min()
		x_stop = x_coords.max() + 1
		sparse_delegate = self.delegate[y_start:y_stop,x_start:x_stop]
		(y_offset, x_offset) = (s.start for s in self.slice)
		sparser_slice = numpy.s_[y_start+y_offset:y_stop+y_offset,x_start+x_offset:x_stop+x_offset]
		return Heatmap(sparse_delegate, sparser_slice, self.shape)

	def as_dense(self):
		dense_delegate = numpy.zeros(self.shape, dtype=numpy.float32)
		dense_delegate[self.slice] = self.delegate
		return Heatmap(dense_delegate)

	def sum(self):
		return self.delegate.sum()

	def negative(self):
		return Heatmap(1 - self.delegate, self.slice, self.shape)

	def reweight(self, weight):
		return Heatmap(self.delegate * weight, self.slice, self.shape)

	def total_variance(self):
		return (self.delegate**2).sum()

	def normalize_sum(self):
		"""The returned heatmap will be normalized so the sum is one"""
		return Heatmap(self.delegate / self.delegate.sum(), self.slice, self.shape)

	def size(self):
		return functools.reduce(operator.mul, self.shape, 1)

	def normalize_stdev(self):
		"""The returned heatmap will be normalized so the standard deviation is one"""
		pixel_mean = self.delegate.sum() / self.size()
		diff_stdev =  math.sqrt(((self.delegate - pixel_mean)**2).sum() / self.size())
		if diff_stdev:
			normalized_delegate = self.delegate / diff_stdev
		else:
			normalized_delegate = self.delegate
		return Heatmap(normalized_delegate, self.slice, self.shape)

	@staticmethod
	def blank(projection_shape):
		return Heatmap(numpy.float32([]).reshape((0,0)), numpy.s_[0:0,0:0], projection_shape)

	@staticmethod
	def blend(pieces):
		return Heatmap.product_ones([piece.negative() for piece in pieces]).negative()

	@staticmethod
	def union(pieces):
		nonempty_pieces = [piece for piece in pieces if any(s.start < s.stop for s in piece.slice)]
		y_start = min(piece.slice[0].start for piece in nonempty_pieces)
		y_stop = max(piece.slice[0].stop for piece in nonempty_pieces)
		x_start = min(piece.slice[1].start for piece in nonempty_pieces)
		x_stop = max(piece.slice[1].stop for piece in nonempty_pieces)
		max_shape = max(piece.shape for piece in nonempty_pieces)
		return ((y_stop - y_start, x_stop - x_start),
			[numpy.s_[piece.slice[0].start-y_start:piece.slice[0].stop-y_start,piece.slice[1].start-x_start:piece.slice[1].stop-x_start]
				for piece in pieces],
			numpy.s_[y_start:y_stop,x_start:x_stop],
			max_shape,
		)

	@staticmethod
	def intersection(pieces):
		y_start = max(piece.slice[0].start for piece in pieces)
		y_stop = min(piece.slice[0].stop for piece in pieces)
		x_start = max(piece.slice[1].start for piece in pieces)
		x_stop = min(piece.slice[1].stop for piece in pieces)
		min_shape = min(piece.shape for piece in pieces)
		if y_stop <= y_start or x_stop <= x_start:
			intersected = ((0, 0), [numpy.s_[0:0,0:0] for _ in pieces], numpy.s_[0:0,0:0], min_shape)
		else:
			intersected = ((y_stop - y_start, x_stop - x_start),
				[numpy.s_[y_start-piece.slice[0].start:y_stop-piece.slice[0].start,x_start-piece.slice[1].start:x_stop-piece.slice[1].start]
					for piece in pieces],
				numpy.s_[y_start:y_stop,x_start:x_stop],
				min_shape,
			)
		return intersected

	@staticmethod
	def product_ones(pieces):
		(broadcast_shape, indices, combined_slice, combined_shape) = Heatmap.union(pieces)
		combined = numpy.ones(broadcast_shape, dtype=numpy.float32)
		for (piece, index) in zip(pieces, indices):
			combined[index] *= piece.delegate
		return Heatmap(combined, combined_slice, combined_shape)

	@staticmethod
	def product_zeros(pieces):
		"""Performs element-wise multiplication and returns the sum"""
		(broadcast_shape, indices, combined_slice, combined_shape) = Heatmap.intersection(pieces)
		broadcasts = [piece.delegate[index] for (piece, index) in zip(pieces, indices)]
		combined = numpy.prod(broadcasts, axis=0)
		return Heatmap(combined, combined_slice, combined_shape)

	def subtract(self, other):
		(broadcast_shape, (index_self, index_other), combined_slice, combined_shape) = Heatmap.union([self, other])
		combined = numpy.zeros(broadcast_shape, dtype=numpy.float32)
		combined[index_self] = self.delegate
		combined[index_other] -= other.delegate
		return Heatmap(combined, combined_slice, combined_shape)

	def subtract_ones(self, other):
		(broadcast_shape, (index_self, index_other), combined_slice, combined_shape) = Heatmap.union([self, other])
		combined = numpy.ones(broadcast_shape, dtype=numpy.float32)
		combined[index_self] = self.delegate
		combined[index_other] -= other.delegate
		return Heatmap(combined, combined_slice, combined_shape)

	def is_overlapping(self, other):
		return numpy.any(Heatmap.product_zeros([self, other]).delegate)

	def correlation(self, other):
		"""Performs element-wise multiplication and returns the sum"""
		return Heatmap.product_zeros([self, other]).sum()


def get_piece_heatmaps(frame_size, projection):
	rotation, jacobian = cv2.Rodrigues(projection.pose.rvec.astype(numpy.float32))
	inv_rotation = rotation.transpose()
	inv_tvec = numpy.dot(inv_rotation, -projection.pose.tvec.astype(numpy.float32))
	inv_pose = numpy.vstack([numpy.hstack([inv_rotation, inv_tvec]), numpy.float32([0, 0, 0, 1])])
	inv_camera_matrix = numpy.linalg.inv(projection.cameraIntrinsics.cameraMatrix.astype(numpy.float32))
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
	piece_voxels = voxels.get_piece_voxels(horizontal_resolution, vertical_resolution)

	# This helper performs volume ray casting
	ctx = ModernGL.create_standalone_context()

	voxels_size = tuple(reversed(piece_voxels.shape))
	texture = ctx.texture3d(voxels_size, 1, piece_voxels, floats=True)
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
				gl_Position = vec4(canvas_coord, 0, 1);
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
		shift = numpy.float32([
			[1, 0, 0, -i],
			[0, 1, 0, -j],
			[0, 0, 1, 0],
			[0, 0, 0, 1],
		])
		stretch = numpy.float32([
			[1, 0, 0, 0],
			[0, 1, 0, 0],
			[0, 0, 1 / height, 0],
			[0, 0, 0, 1],
		])
		piece_inv_pose = numpy.dot(numpy.dot(shift, stretch), inv_pose)
		piece_inv_projection = numpy.dot(piece_inv_pose, ext_inv_camera_matrix)
		camera_position = numpy.dot(piece_inv_pose, numpy.float32([0, 0, 0, 1]).reshape(4,1))

		prog.uniforms['inv_projection'].write(piece_inv_projection[:3].transpose())
		prog.uniforms['camera_position'].write(camera_position[:3])
		prog.uniforms['piece_dimensions'].write(numpy.float32([1, 1, height]))

		ctx.clear()
		# TODO: It would be possible to speed this up by shrinking the frame
		# until it barely contains the bounding box
		vao.render(ModernGL.TRIANGLE_STRIP)

		data = fbo.read(components=1, floats=True)
		heatmap = numpy.frombuffer(data, dtype=numpy.float32).reshape(projection_shape)

		# TODO: Optimize the raycaster so it only creates the relevant part of the heatmap
		heatmaps[(square, piece_type)] = Heatmap(heatmap).as_sparse()

	return heatmaps


def get_reference_heatmap(heatmaps):
	all_kings_heatmaps = (heatmaps[(square, chess.KING)] for square in chess.SQUARES)
	return Heatmap.blend(all_kings_heatmaps).normalize_sum()


def get_board_heatmap(heatmaps, board):
	pieces = board.piece_map().items()
	piece_heatmaps = (heatmaps[(square, piece.piece_type)] for (square, piece) in pieces)
	return Heatmap.blend(piece_heatmaps)


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


def get_occlusions(heatmaps, projection):
	depths = get_depths(projection)
	occlusions = collections.defaultdict(set)
	for (square_a, square_b) in itertools.product(chess.SQUARES, repeat=2):
		if square_a == square_b:
			continue
		(piece_a, piece_b) = [heatmaps[(square, chess.KING)] for square in (square_a, square_b)]
		if not piece_a.is_overlapping(piece_b):
			continue

		ordered_squares = sorted([square_a, square_b], key=lambda square: depths[square])
		occlusions[ordered_squares[1]].add(ordered_squares[0])
	return collections.defaultdict(list,
		((square, sorted(values, key=lambda square: depths[square])) for (square, values) in occlusions.items()))


def get_negative_composite(negative_composite_memo, heatmaps, sorted_pieces):
	if sorted_pieces in negative_composite_memo:
		#print('        CACHE HIT ', sorted_pieces)
		return negative_composite_memo[sorted_pieces]
	#print('        CACHE MISS', sorted_pieces)

	(sorted_pieces_tail, sorted_pieces_head) = (sorted_pieces[:-1], sorted_pieces[-1])
	negative_composite_tail = get_negative_composite(negative_composite_memo, heatmaps, sorted_pieces_tail)
	negative_composite_head = heatmaps[sorted_pieces_head]
	negative_composite = Heatmap.product_ones([negative_composite_tail, negative_composite_head.negative()])
	negative_composite_memo[sorted_pieces] = negative_composite
	return negative_composite


def get_piece_diff(negative_composite_memo, heatmaps, occlusions, stable_pieces, focal_piece):
	stable_piece_map = dict(stable_pieces)
	occluding_pieces = tuple((occluding_square, stable_piece_map[occluding_square])
		for occluding_square in occlusions[focal_piece[0]]
		if occluding_square in stable_piece_map)
	negative_composite = get_negative_composite(negative_composite_memo, heatmaps, occluding_pieces)
	negative_layer = heatmaps[focal_piece].negative()
	combined_negative_composite = Heatmap.product_ones([negative_composite, negative_layer])
	diff = negative_composite.subtract_ones(combined_negative_composite)
	return diff


def get_move_diffs(heatmaps, reference_heatmap, occlusions, negative_composite_memo, board):
	move_diffs = {}
	for move in board.legal_moves:
		pieces_before = board.piece_map().items()
		board.push(move)
		try:
			pieces_after = board.piece_map().items()
		finally:
			board.pop()
		moved_piece_items = pieces_before ^ pieces_after
		moved_pieces = frozenset([(square, piece.piece_type) for (square, piece) in moved_piece_items])
		stable_pieces = [(square, piece.piece_type)
			for (square, piece) in board.piece_map().items()
			if (square, piece) not in moved_pieces]
		piece_diffs = (get_piece_diff(negative_composite_memo, heatmaps, occlusions, stable_pieces, piece_item)
			for piece_item in moved_pieces)

		combined_diff = Heatmap.blend(piece_diffs)
		normalized_diff = combined_diff.as_sparse().normalize_stdev()
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

	heatmaps = get_piece_heatmaps(frame_size, projection)
	reference_heatmap = get_reference_heatmap(heatmaps)
	occlusions = get_occlusions(heatmaps, projection)
	board = chess.Board()
	projection_shape = tuple(reversed(frame_size))
	negative_composite_memo = {(): Heatmap.blank(projection_shape)}

	subtractor = cv2.imread('diff.png')[:,:,0]
	normalized_subtractor = subtractor / subtractor.stdev()

	move_diffs = get_move_diffs(heatmaps, reference_heatmap, occlusions, negative_composite_memo, board)
	for (move, move_diff) in move_diffs.items():
		# The Pearson correlation coefficient measures the goodness of fit
		score = (move_diff * normalized_subtractor).mean()
		print('score', board.san(move), score)
		composite = numpy.zeros((720, 1280, 3), dtype=numpy.float32)
		composite[:,:,2] += move_diff
		composite[:,:,1] += normalized_subtractor
		cv2.imshow(WINNAME, composite / 20)
		key = cv2.waitKey(0)


if __name__ == "__main__":
	main()
