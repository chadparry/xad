#!/usr/bin/env python3

import collections
import colorsys
import cv2
import itertools
import math
import numpy
import random
import scipy.optimize
import scipy.spatial
import shapely.geometry.polygon
import skimage.measure
import skimage.transform
import sklearn.base
import sklearn.linear_model
import sys
import traceback

import pose


WINNAME = 'Chess Transcription'


def grouper(iterable, n):
	args = [iter(iterable)] * n
	return list(zip(*args))

color_global = None

def main():

	cv2.namedWindow(WINNAME)

	#webcam = cv2.VideoCapture(0)
	#webcam = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')
	webcam = cv2.VideoCapture('idaho.webm')
	if not webcam.isOpened():
		raise RuntimeError('Failed to open camera')

	#webcam.set(cv2.CAP_PROP_POS_MSEC, 1000)
	webcam.set(cv2.CAP_PROP_POS_MSEC, 32000)
	#webcam.set(cv2.CAP_PROP_POS_MSEC, 90000)

	pattern_size = (7, 7)
	(retval, color2) = webcam.read()

	img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	#cv2.imshow(WINNAME, color2)
	#key = cv2.waitKey(0)
	global color_global
	color_global = numpy.copy(color2)

	#(retval, color1) = webcam.read()
	color1 = numpy.copy(color2)

	#cv2.imshow(WINNAME, img1)
	#key = cv2.waitKey(0)

	corners = find_chessboard_corners(color2)
	projection = get_projection(corners, color2.shape)

	chess_grid = numpy.array([[[float(x), float(y), 0.] for y in range(9) for x in range(9)]])
	project_grid_points_result, j = cv2.projectPoints(chess_grid, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	project_grid_points = project_grid_points_result.reshape(9, 9, 2)

	pimg = numpy.copy(color2)
	square = numpy.int32([[
		project_grid_points[0][0],
		project_grid_points[-1][0],
		project_grid_points[-1][-1],
		project_grid_points[0][-1],
	]])
	cv2.fillPoly(pimg, square, (255, 255, 255))
	for y in range(8):
		for x in range(8):
			is_light = bool((x + y) % 2)
			if is_light:
				continue
			square = numpy.int32([[
				project_grid_points[y][x],
				project_grid_points[y+1][x],
				project_grid_points[y+1][x+1],
				project_grid_points[y][x+1],
			]])
			cv2.fillPoly(pimg, square, (0, 0, 0))

	axis = numpy.float32([[0, 0, 0], [4,0,0], [0,4,0], [0,0,4]]).reshape(-1,3)
	imgpts, jac = cv2.projectPoints(axis, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	for pt in imgpts[1:]:
		cv2.line(pimg, tuple(imgpts[0].ravel()), tuple(pt.ravel()), (0,255,0), 2)


	square_size_mm = 57.15
	king_height = 95. / square_size_mm
	queen_height = 85. / square_size_mm
	bishop_height = 70. / square_size_mm
	knight_height = 60. / square_size_mm
	rook_height = 55. / square_size_mm
	pawn_height = 50. / square_size_mm

	pieces_coord = numpy.array([[float(x) + 0.5, 0.5, float(z)] for x in range(0, 8) for z in [0., queen_height]])
	project_grid_points_result, j = cv2.projectPoints(pieces_coord, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	project_grid_points = project_grid_points_result.reshape(8, 2, 2)

	for pidx in range(8):
		base = project_grid_points[pidx][0]
		top = project_grid_points[pidx][1]
		cv2.line(pimg, tuple(base.astype('int32')), tuple(top.astype('int32')), (255,192,192), 15)

	pieces_coord = numpy.array([[float(x) + 0.5, 1.5, float(z)] for x in range(0, 8) for z in [0., pawn_height]])
	project_grid_points_result, j = cv2.projectPoints(pieces_coord, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	project_grid_points = project_grid_points_result.reshape(8, 2, 2)

	for pidx in range(8):
		base = project_grid_points[pidx][0]
		top = project_grid_points[pidx][1]
		cv2.line(pimg, tuple(base.astype('int32')), tuple(top.astype('int32')), (255,192,192), 15)


	pieces_coord = numpy.array([[float(x) + 0.5, 7.5, float(z)] for x in range(0, 8) for z in [0., queen_height]])
	project_grid_points_result, j = cv2.projectPoints(pieces_coord, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	project_grid_points = project_grid_points_result.reshape(8, 2, 2)

	for pidx in range(8):
		base = project_grid_points[pidx][0]
		top = project_grid_points[pidx][1]
		cv2.line(pimg, tuple(base.astype('int32')), tuple(top.astype('int32')), (128,0,0), 15)

	pieces_coord = numpy.array([[float(x) + 0.5, 6.5, float(z)] for x in range(0, 8) for z in [0., pawn_height]])
	project_grid_points_result, j = cv2.projectPoints(pieces_coord, projection.pose.rvec, projection.pose.tvec, projection.cameraIntrinsics.cameraMatrix, projection.cameraIntrinsics.distCoeffs)
	project_grid_points = project_grid_points_result.reshape(8, 2, 2)

	for pidx in range(8):
		base = project_grid_points[pidx][0]
		top = project_grid_points[pidx][1]
		cv2.line(pimg, tuple(base.astype('int32')), tuple(top.astype('int32')), (128,0,0), 15)


	cv2.imshow(WINNAME, pimg)
	key = cv2.waitKey(0)


def find_chessboard_corners(image):
	color1 = numpy.copy(image)
	color2 = numpy.copy(image)

	img1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)

	#ret,thresh = cv2.threshold(img1,127,255,0)
	contours = []
	is_light_contours = []
	# Start off with the largest reasonable block size, which should be near the smallest
	# possible dimension that four chessboard squares could have. That way, at the edge
	# of the board, the block size could still encompass the two nearest ranks or files.
	# This loop starts with a block size of 31, meaning that each square is expected to be
	# at least 8 pixels high.
	for block_size in (2**exp - 1 for exp in itertools.count(5)):
		if block_size * 2 >= min(img1.shape[0:2]):
			break
		thresh = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,block_size,0)
		#cv2.imshow(WINNAME, thresh)
		#key = cv2.waitKey(0)

		for dilation in [4, 1]:

			kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(dilation, dilation))

			# For finding dark squares
			dilated = cv2.dilate(thresh, kernel)

			# For finding light squares
			eroded = cv2.erode(thresh, kernel)

			# Draw a rectangle around the outer edge,
			# so that clipped corners have a chance of being recognized.
			cv2.rectangle(dilated, (0, 0), (dilated.shape[1]-1, dilated.shape[0]-1), 255)
			cv2.rectangle(eroded, (0, 0), (eroded.shape[1]-1, eroded.shape[0]-1), 0)

			#cv2.imshow(WINNAME, dilated)
			#key = cv2.waitKey(0)
			#cv2.imshow(WINNAME, eroded)
			#key = cv2.waitKey(0)

			im2, contoursd, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			ime2, contourse, hierarchy = cv2.findContours(eroded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

			contoursd_rev = [numpy.array(list(reversed(contour))) for contour in contoursd]

			filtered_contoursd = filter_quads(contoursd_rev)
			filtered_contourse = filter_quads(contourse)

			kernel_hull = get_dilated_kernel_hull(kernel)
			dilated_contoursd = dilate_contours(filtered_contoursd, kernel_hull)
			dilated_contourse = dilate_contours(filtered_contourse, kernel_hull)

			contours.extend(dilated_contoursd)
			is_light_contours.extend(False for _ in dilated_contoursd)
			contours.extend(dilated_contourse)
			is_light_contours.extend(True for _ in dilated_contourse)



	#print('contours', len(contours))

	# Filter out duplicate quads
	tree = scipy.spatial.KDTree([corner for contour in contours for corner in contour])
	dist = numpy.linalg.norm([1, 1])
	pairs = tree.query_pairs(dist)
	#print('pairs', pairs)
	connected_indices = collections.defaultdict(lambda: collections.defaultdict(list))
	for pair in pairs:
		quad_idxs = [p // 4 for p in pair]
		if quad_idxs[0] == quad_idxs[1]:
			continue
		for (left, right) in itertools.permutations(quad_idxs):
			connected_indices[left][right].append(pair)
	#print('CONN_IND', len(connected_indices), connected_indices)
	# If two quads overlap on 3 or more corners, then discard the one that was found
	# first, which is the one found in the noiser image with the smaller block size and larger kernel size.
	complementary = {left_idx: right_counts
		for (left_idx, right_counts) in connected_indices.items()
		if any(
				all(is_complementary_corner(pair[0], pair[1], contours, color2) for pair in connected_pairs)
				for (right_idx, connected_pairs) in right_counts.items())}
	uniques = {left_idx: right_counts
		for (left_idx, right_counts) in complementary.items()
		if all(
			right_idx < left_idx or
			right_idx not in complementary or
			len(connected_pairs) < 3
			for (right_idx, connected_pairs) in right_counts.items())}


	#print('UNIQUES', len(uniques), uniques)
	connected_map = [left_idx for (left_idx, right_keys) in uniques.items() if any(key in uniques for key in right_keys)]
	#print('CONN_MAP', len(connected_map), connected_map)
	reverse_connected_map = {old_idx: new_idx for (new_idx, old_idx) in enumerate(connected_map)}
	quads = [contours[left_idx] for left_idx in connected_map]
	is_light_quads = [is_light_contours[left_idx] for left_idx in connected_map]
	print('quads', len(quads))

	contlines = numpy.zeros((color2.shape[0], color2.shape[1], 3), numpy.uint8)
	#for contour in contours:
	#	cv2.drawContours(contlines, numpy.array([[(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1)))
	#	for (x,y) in contour]]), -1, (0, 0, 255), 1)
	for contour in quads:
		cv2.drawContours(contlines, numpy.array([[(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1)))
		for (x,y) in contour]]), -1, (255, 0, 0), 1)
	#cv2.imshow(WINNAME, contlines)
	#key = cv2.waitKey(0)

	# TODO: Once the board is found, use MSER to track it

	color3 = numpy.copy(color2)

	# FIXME: Discard any RANSAC sample sets where the furthest points are more than 9 units apart.
	class ChessboardPerspectiveEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
		def __init__(self, tol=math.pi/36000., shape=(1,1), seed=None):
			self.set_params(tol=tol, shape=shape, seed=seed)
			self.vps = None
		def segment_alignment_error(self, segment, vp, debug=False):
			# FIXME: Isn't the angle the same even if the coordinates are unsorted?
			oriented = sorted(segment, key=lambda point: numpy.linalg.norm(point - vp), reverse=True)
			(closer, further) = oriented
			direction = closer - further
			vanisher = vp - further
			angle = angle_between(direction, vanisher)
			if debug:
				print('angle', angle, list(direction), list(vanisher))
			return abs(angle)
		def quad_alignment_error(self, quad, vp, debug=False):
			return (self.segment_alignment_error(numpy.array(quad[:2]), vp, debug) +
				self.segment_alignment_error(numpy.array(quad[2:]), vp, debug))
			# TODO:
			# Calculate the error due to the nearest point needing a shift so that
			# the angle points exactly toward the vanishing point.
			# Calculate the error due to the ratios not being equal:
			#  The distance to the near point divided by the distance to the far point
			#  should be equal for all lines of all quads to their respective vanishing points.
			# Find a translation that matches all quads and calculate the error
			# due to translating to a chess square of the correct color.
			# In fit(), calculate the homography corresponding to those points.
			# Alternatively, add a boolean to each quad to signify its color, then
			# all y values can be zero, and the distance is returned by predict().

			return distance
		def quad_error(self, vps, quad):
			(vp1, vp2) = vps
			horizon = vp2 - vp1
			horizon_norm = numpy.linalg.norm(horizon)
			unit_horizon = horizon / horizon_norm if horizon_norm != 0. else numpy.zeros(horizon.shape)
			distance = 0.

			projected_quad = []
			for point in quad:
				# Check which side of the horizon each point is on.
				if numpy.linalg.det([horizon, point - vp1]) < 0:
					# Move any points above the horizon directly onto the horizon
					projection = numpy.dot(point - vp1, unit_horizon)
					projected_point = vp1 + projection * unit_horizon
					# FIXME: This isn't the same scale as the angular error
					adjustment = numpy.linalg.norm(projected_point - point)**2
					#print('adjustment', adjustment)
					distance += adjustment
				else:
					projected_point = point
				projected_quad.append(projected_point)

			rotated_quad = projected_quad[1:] + projected_quad[:1]
			distance += self.quad_alignment_error(projected_quad, vp1) + self.quad_alignment_error(rotated_quad, vp2)

			#print('distance', distance)
			return distance

		def objective(self, sample_quads, x, wait=False):
			(vp1, vp2) = vps = [numpy.array(vp) for vp in grouper(x, 2)]
			#print('vp', vp1, vp2)

			distance = sum(self.quad_error(vps, quad) for quad in sample_quads)
			return distance

			#print('distance', distance)

			working = numpy.copy(color3)
			left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
			right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
			cv2.drawContours(working, numpy.array(sample_quads), -1, 255, 2)
			cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
			#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
			#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			for quad in sample_quads:
				for point in quad:
					for vp in vps:
						cv2.line(working, point, tuple(int(d) for d in vp), (0,0,255), 1)

			#cv2.imshow(WINNAME, working)
			#key = cv2.waitKey(1 if wait else 0)

			return distance
		def fit(self, X, y):
			sample_quads = [grouper(quad, 2) for quad in X]

			#print('optimizing...')
			optimizer = lambda x: self.objective(sample_quads, x)

			#res = scipy.optimize.basinhopping(optimizer, self.seed, minimizer_kwargs={ 'method': 'L-BFGS-B', 'options': { 'ftol': 1e-2 } })
			#if not res.lowest_optimization_result.success:
			#	raise RuntimeError('solver failed: ' + res.lowest_optimization_result.message)
			#fitted = res.lowest_optimization_result.x

			#res = scipy.optimize.differential_evolution(optimizer, [(-1., 1) for s in self.seed])
			#if not res.success:
			#	raise RuntimeError('solver failed: ' + res.message)
			#fitted = res.x

			#res = scipy.optimize.root(lambda x: [optimizer(x)] + [0]*(len(x)-1), self.seed, method='lm')
			tol = self.get_params()['tol']
			provided_seed = self.get_params()['seed']
			if provided_seed is None:
				shape = self.get_params()['shape']
				seed = [shape[1]/2., shape[0]/2., float(shape[1]), shape[0]/2.]
			else:
				seed = [dim for vp in provided_seed for dim in vp]

			#res = scipy.optimize.minimize(lambda x: optimizer(x), seed, method='L-BFGS-B', tol=tol)
			#if not res.success:
			#	self.vps = None
			#	return self
			#fitted = res.x
			#self.vps = numpy.array(grouper(fitted, 2))
			#self.vps = get_vps(sample_quads)


			self.vps = get_best_intersection_by_angle5_quads(sample_quads, tol)
			delay = 1
			if self.vps is None:
				return self



			(vp1, vp2) = self.vps
			if any(dim > 1e9 or dim < -1e9 for vp in self.vps for dim in vp):
				print('overflow', self.vps)
				return self
			working = numpy.copy(color3)
			cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in sample_quad]) for sample_quad in sample_quads]), -1, 255, 2)
			cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
			cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
			cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 2)
			for quad in sample_quads:
				for point in quad:
					for vp in self.vps:
						cv2.line(working, tuple(int(dim) for dim in point), tuple(int(d) for d in vp), (0,0,255), 1)

			cv2.imshow(WINNAME, working)
			key = cv2.waitKey(delay)

			if delay == 0:
				self.vps = None
			#print('comparison', self.objective(sample_quads, [360., 200., 6000., 240.], True))

			# FIXME
			# The translation had already been determined deep inside the objective function,
			# but it was lost and needs to be recovered.

			return self
		def score(self, X, y):
			#s = super(ChessboardPerspectiveEstimator, self).score(X, y)
			s = 1 - sum(cost[0] ** 2 for cost in self.predict(X))
			print('score', s, self.vps, len(X))
			#print('SCORE', X, y, s)
			return s
		def predict(self, X):
			#print('predict', self.vps, len(X))
			quads = [grouper(quad, 2) for quad in X]
			if self.vps is None:
				return [float('inf') for quad in quads]
			predicted = [(get_max_error_by_angle5_quad(self.vps, quad),) for quad in quads]
			scores = sorted(s[0]**2 for s in predicted)
			print('predict', len(X), scores[:3], scores[-3:])
			#print('PREDICT', X, predicted)
			return predicted
		def rotate_quad(self, quad, tq=None, debug=False):
			"""
			Transform the points by an integral distance and
			reorder the points so the upper-leftmost is first
			"""
			#center = shapely.geometry.polygon.Polygon(quad).representative_point()
			center = get_centroid(quad)
			# Start with the x-coordinate
			#(transformx, transformy) = (-((center[0] -
			#	# Then decide whether to shift by one square, so the color doesn't change
			#	center[1] % 2 // 1
			#	# Then find which group of two squares the center is in
			#	) // 2 * 2),
			#	# The x coord already preserves the color, so the y coord doesn't need to shift.
			#	-(center[1] // 1))

			# First find the distance to the origin
			(transformx, transformy) = (-(center[0] // 1 -
				# Then decide whether to shift by one square, so the color doesn't change
				(center[0] % 2 // 1 + center[1] % 2 // 1) % 2),
				# The x coord already preserves the color, so the y coord doesn't need to shift.
				-(center[1] // 1))
			transformed = [(x + transformx, y + transformy) for (x, y) in quad]

			angles = (math.atan2(y - center[1], x - center[0]) for (x, y) in quad)
			if debug or True:
				angles = list(angles)
			# It is assumed the points are already listed counter-clockwise and their sequence should be preserved.
			# If the homography inverted the orientation, then the error measurement will be high
			# and the homography will need to be discarded.
			# Each angle should be 90 degrees greater than the previous angle. All the angles will
			# have the same bias (angular distance from the desired orientation) if the quad is a perfect square.
			bias_angles = (angle + i * (math.pi/2.) for (i, angle) in enumerate(angles))
			if debug or True:
				bias_angles = list(bias_angles)
			average_bias_angle = math.atan2(
				sum(math.sin(angle) for angle in bias_angles),
				sum(math.cos(angle) for angle in bias_angles))
			# Reorder the points to minimize the bias.
			rotation = int(average_bias_angle / (math.pi/2.) + 2)
			#print('rotation', rotation)
			rotated = transformed[rotation:] + transformed[:rotation]

			if tq is not None:
				score = sum((td-rd)**2 for (rp, tp) in zip(rotated, tq) for (rd, td) in zip(rp, tp))
				#if score > 2:
				#	debug = True

			if debug:
				print('--------------------------------')
				print('center', center)
				#print('shifted', (shiftedx, shiftedy))
				print('transform', (transformx, transformy))
				print('new center', (center[0] + transformx, center[1] + transformy))
				print('offsets', [(x - center[0], y - center[1]) for (x, y) in quad])
				print('raw angles', [a / (math.pi/2.) for a in angles])
				print('bias angles', [a / (math.pi/2.) for a in bias_angles])
				print('average bias angle', average_bias_angle / (math.pi/2.))
				print('rotation', rotation)
				print('rotated from', transformed)
				print('rotated to', rotated)
				if center[0] + transformx >= 2:
					print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
			return rotated

	if len(quads) < 3:
		raise RuntimeError('Not enough quads found')

	# RANSACRegressor expects the input to be an array of points.
	# This target data is an array of quads instead, where each quad
	# contains 4 points. The translation is done by passing all 4 2-D
	# points as if they were a single 8-dimensional point.
	quads_rot = [numpy.concatenate([quad[1:], quad[:1]]) for quad in quads]
	target_pts = [[dim for corner in quad for dim in corner] for quad in quads + quads_rot]
	dark_square = (0., 0., 0., 1., 1., 1., 1., 0.)
	light_square = (1., 0., 1., 1., 2., 1., 2., 0.)
	training_pts = [(0,) for i in range(len(target_pts))]

	threshold = sum((dim/16.)**2 for dim in color3.shape[0:2])
	#print('thresh', threshold)
	visible_squares_estimate = 3
	success_rate = 0.999999
	retries = int(math.ceil(math.log(1 - success_rate,
		max(0.5, 1 - visible_squares_estimate / float(len(target_pts))))))
	while retries > 0:
		regressor = sklearn.linear_model.RANSACRegressor(
			base_estimator=ChessboardPerspectiveEstimator(tol=math.pi/360., shape=color3.shape),
			min_samples=3,
			residual_metric=lambda dy: numpy.sum(dy**2, axis=1),
			# FIXME: Each segment has only 3 degrees of allowed error on average.
			residual_threshold=(math.pi/120.)**2,
			#residual_threshold=threshold,
			max_trials=retries,
		)
		#regressor.get_params()['base_estimator'].set_params(tol=math.pi/360., shape=color3.shape)
		try:
			regressor.fit(target_pts, training_pts)
			break
		except ValueError as e:
			#print('Failed regression', e)
			try:
				retries -= regressor.n_trials_
			except AttributeError:
				retries = 0
			if retries <= 0:
				raise
	vps = regressor.estimator_.vps
	if vps is None:
		raise RuntimeError('Inliers not found')

	print('retries', retries, regressor.n_trials_)
	(vp1, vp2) = vps
	working = numpy.copy(color3)
	left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
	right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
	# FIXME: Use itertools.compress
	inlier_quads = [quad for (quad, mask) in zip(quads + quads_rot, regressor.inlier_mask_) if mask]
	#print('vps', tuple(vp1), tuple(vp2))
	#print('quads', inlier_quads)
	try:
		cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in quad]) for quad in inlier_quads]), -1, 255, 2)
		cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
		#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
		#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
		cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
		#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
		#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
		cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 2)
		for quad in inlier_quads:
			for point in quad:
				for vp in [vp1, vp2]:
					cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)
	except OverflowError:
		pass

	inlier_indices = [idx % len(quads) for (idx, mask) in enumerate(regressor.inlier_mask_) if mask]
	# Discard quads that are no longer touching any other inliers
	filtered_inlier_indices = [idx for idx in inlier_indices
		if any(right_idx in reverse_connected_map and reverse_connected_map[right_idx] in inlier_indices for right_idx in uniques[connected_map[idx]])]
	#inlier_quads = [quads[idx] for idx in filtered_inlier_indices]
	# FIXME: Scanning the quads should not be necessary to filter.
	#inlier_quads = [quad for (idx, quad) in enumerate(quads + quads_rot)
	#	if regressor.inlier_mask_[idx] and idx % len(quads) in filtered_inlier_indices]
	inlier_quads_filter = [regressor.inlier_mask_[idx] and idx % len(quads) in filtered_inlier_indices for idx in range(len(quads) + len(quads_rot))]
	inlier_quads = list(itertools.compress(quads + quads_rot, inlier_quads_filter))
	is_light_inlier_quads = list(itertools.compress(is_light_quads * 2, inlier_quads_filter))
	cv2.imshow(WINNAME, working)
	key = cv2.waitKey(1)
	if len(inlier_quads) < 3:
		raise RuntimeError('Not enough quads found')

	# Run it again on the inliers with higher precision
	# FIXME: Now that the inliers and edge orientations are known, an analytic solution is possible
	# See http://math.stackexchange.com/questions/61719/finding-the-intersection-point-of-many-lines-in-3d-point-closest-to-all-lines
	# http://www.mathworks.com/matlabcentral/fileexchange/37192-intersection-point-of-lines-in-3d-space
	estimator = ChessboardPerspectiveEstimator(tol=math.pi/360000., shape=color3.shape, seed=(vp1, vp2))
	target_pts = [[dim for corner in quad for dim in corner] for quad in inlier_quads]
	training_pts = [(0,) for i in range(len(inlier_quads))]
	estimator.fit(target_pts, training_pts)
	(vp1, vp2) = estimator.vps

	try:
		working = numpy.copy(color3)
		left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
		right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
		#inlier_quads = [quad for (quad, mask) in zip(inlier_quads, regressor.inlier_mask_) if mask]
		#print('quads', len(inlier_quads))
		cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in quad]) for quad in inlier_quads]), -1, 255, 2)
		cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
		#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
		#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
		cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
		#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
		#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
		cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 2)
		for quad in inlier_quads:
			for point in quad:
				for vp in [vp1, vp2]:
					cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)
	except OverflowError:
		pass
	cv2.imshow(WINNAME, working)
	key = cv2.waitKey(1)


	inlier_quads = numpy.array(inlier_quads)
	# Calculate the camera parameters
	# See https://fedcsis.org/proceedings/2012/pliks/110.pdf
	horizon = vp2 - vp1
	horizon_norm = numpy.linalg.norm(horizon)
	unit_horizon = horizon / horizon_norm if horizon_norm != 0. else numpy.zeros(horizon.shape)
	image_dim = numpy.array([color3.shape[1], color3.shape[0]])
	oi = image_dim / 2.

	oi_projection1 = numpy.dot(oi - vp1, unit_horizon)
	oi_projection2 = numpy.dot(oi - vp2, unit_horizon)
	# Vanishing points near infinity cause large rounding errors, so we choose the shortest projection
	if oi_projection1 <= oi_projection2:
		vi = vp1 + oi_projection1 * unit_horizon
	else:
		vi = vp2 + oi_projection2 * unit_horizon

	# The horizon may be inverted, which would result in a homography that looks through the bottom of the chessboard
	# This can be corrected by switching the vanishing points
	if numpy.linalg.det([vi - oi, vp2 - vp1]) < 0:
		(vp1, vp2) = (vp2, vp1)

	#print('vi', vi)
	square_f = numpy.linalg.norm(vi - vp1) * numpy.linalg.norm(vi - vp2) - numpy.linalg.norm(vi - oi)**2
	if square_f > 0:
		f = math.sqrt(square_f)
		print('focal length', f)
	else:
		print("Camera center is not aligned with image center", file=sys.stderr)
		f = max(image_dim)
	#print('f', f)

	# FIXME
	#vp1 = (1537.3, -6.2)
	#vp2 = (402.6, 2520.5)
	#f = 1

	x_axis = numpy.array([vp1[0], vp1[1], f])
	y_axis = numpy.array([vp2[0], vp2[1], f])
	unit_x_axis = x_axis / numpy.linalg.norm(x_axis)
	unit_y_axis = y_axis / numpy.linalg.norm(y_axis)
	unit_z_axis = numpy.cross(unit_x_axis, unit_y_axis)
	denom1 = math.sqrt(vp1[0]**2 + vp1[1]**2 + f)
	denom2 = math.sqrt(vp2[0]**2 + vp2[1]**2 + f)
	#print('x', unit_x_axis)
	#print('y', unit_y_axis)
	#print('z', unit_z_axis)
	rdenorm = numpy.array([
		[vp1[0] / denom1, vp2[0] / denom2, unit_z_axis[0]],
		[vp1[1] / denom1, vp2[1] / denom2, unit_z_axis[1]],
		[f / denom1, f / denom2, unit_z_axis[2]],
	])
	r = rdenorm / rdenorm[2][2]
	#print('R', r)



	(h, w, _) = color3.shape
	f = max(h, w)
	fx = fy = f
	#(fx, fy) = (120., 16.)
	default_mtx = numpy.array([[fx, 0, (w - 1)/2.], [0, fy, (h - 1)/2.], [0, 0, 1]]).astype('float32')
	inv_default_mtx = numpy.linalg.inv(default_mtx)
	x_axis = numpy.array([vp1[0], vp1[1], 1])
	y_axis = numpy.array([vp2[0], vp2[1], 1])
	r1_denorm = numpy.dot(inv_default_mtx, x_axis)
	r2_denorm = numpy.dot(inv_default_mtx, y_axis)
	r1 = r1_denorm / numpy.linalg.norm(r1_denorm)
	r2 = r2_denorm / numpy.linalg.norm(r2_denorm)
	r3 = numpy.cross(r1, r2)
	rdenorm = numpy.concatenate([r1.reshape(3,1), r2.reshape(3,1), r3.reshape(3,1)], axis=1)
	r = rdenorm / rdenorm[2][2]
	dist = None

	tvec = numpy.array([0., 0., 1.])
	# Huge squares:
	#tvec = numpy.array([-1., 0., 0.])
	#zoom_out = numpy.array([[0.05, 0., 0.], [0., 0.1, 0.], [0., 0., 1.]])
	#tvec = numpy.array([-10., 10., 0.])
	#zoom_out = numpy.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
	#tvec = numpy.array([-100., 50., 0.])
	#zoom_out = numpy.array([[3., 0., 0.], [0., 10., 0.], [0., 0., 1.]])

	#r = numpy.dot(r, zoom_out)
	#r = r / r[2][2]

	#print('R', r)
	#print('tvec', tvec.reshape(3,1))

	#rt1 = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array([1., 0., 0., 0.]).reshape(4,1))
	#rt2 = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array([0., 1., 0., 0.]).reshape(4,1))
	#print('round trip 1', vp1, '->', (rt1.reshape(1,3) / rt1[2][0])[0][0:2])
	#print('round trip 2', vp2, '->', (rt2.reshape(1,3) / rt2[2][0])[0][0:2])
	def project(p):
		proj = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array(p).reshape(4,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]
	#print('round trip 1', vp1, '->', project([1., 0., 0., 0.]))
	#print('round trip 2', vp2, '->', project([0., 1., 0., 0.]))

	projectable_quads = numpy.array([[[x+dx, y+dy, 0.] for (dx,dy) in [(0,0),(1,0),(1,1),(0,1)]] for (x,y) in itertools.product(range(8), range(8))])
	bg = numpy.copy(color3)
	for pq in projectable_quads:
		quadpts, j = cv2.projectPoints(pq, r, tvec, default_mtx, dist)
		quadpts = numpy.array([[int(dim) for dim in pt[0]] for pt in quadpts])
		#print('projected quad', quadpts)
		cv2.drawContours(bg, [quadpts], -1, (255, 0, 0), 2)

		quadpts = numpy.array([project([pt[0], pt[1], 0., 1.]) for pt in pq]).astype('int')
		#print('projected quad', quadpts)
		cv2.drawContours(bg, [quadpts], -1, (0, 0, 255), 1)

	try:
		cv2.circle(bg, tuple(int(p) for p in vp1), 5, (0, 255, 0))
		cv2.circle(bg, tuple(int(p) for p in vp2), 5, (0, 255, 0))
		cv2.line(bg, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 2)
	except OverflowError:
		pass
	#cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(1)

	tvec = numpy.array([0., 0., 1.])
	def reverse_project(p):
		#proj = numpy.dot(numpy.linalg.inv(numpy.dot(default_mtx, numpy.concatenate([numpy.delete(r, 2, 1), tvec.reshape(3,1)], axis=1))), numpy.array(p).reshape(3,1))
		proj = numpy.dot(numpy.linalg.inv(numpy.dot(default_mtx, r)), numpy.array(p).reshape(3,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]

	inlier_points_flat = numpy.float32(inlier_quads.reshape(inlier_quads.shape[0] * inlier_quads.shape[1], 2))
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
	accurate_points = cv2.cornerSubPix(img1, inlier_points_flat, (5,5), (-1,-1), criteria)
	accurate_quads = accurate_points.reshape(*inlier_quads.shape)

	rvecs, j = cv2.Rodrigues(r)
	tvecs = tvec.reshape(3, 1)




	bg = numpy.copy(color3)
	projected_quads = []
	for quad in inlier_quads:
		for cu in quad:
			cv2.circle(bg, (int(round(cu[0])), int(round(cu[1]))), 2, (255, 0, 0))
	for quad in accurate_quads:
		quadpts = numpy.array([reverse_project([pt[0], pt[1], 1.]) for pt in quad])
		projected_quads.append(quadpts)
		cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		for cu in quad:
			cv2.circle(bg, (int(round(cu[0])), int(round(cu[1]))), 3, (0, 255, 0))
	#cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(0)

	x_perimeters = numpy.array([get_perimeter(numpy.array([(x, 0) for (x, y) in quad])) for quad in projected_quads])
	y_perimeters = numpy.array([get_perimeter(numpy.array([(0, y) for (x, y) in quad])) for quad in projected_quads])
	outlier_x_perimeter_filter = identify_outliers(x_perimeters)
	outlier_y_perimeter_filter = identify_outliers(y_perimeters)
	outlier_perimeter_filter = [all(f) for f in zip(outlier_x_perimeter_filter, outlier_y_perimeter_filter)]
	filtered_x_perimeters = x_perimeters[outlier_x_perimeter_filter]
	filtered_y_perimeters = y_perimeters[outlier_y_perimeter_filter]
	avg_x_side = sum(filtered_x_perimeters) / float(len(filtered_x_perimeters) * 2)
	avg_y_side = sum(filtered_y_perimeters) / float(len(filtered_y_perimeters) * 2)
	filtered_quads = numpy.array(projected_quads)[outlier_perimeter_filter]
	unprojected_quads = numpy.array(accurate_quads)[outlier_perimeter_filter]
	is_light_filtered_quads = numpy.array(is_light_inlier_quads)[outlier_perimeter_filter]
	#print('AVG_SIDE', avg_x_side, avg_y_side)

	unprojected_pts_with_outliers = unprojected_quads.reshape((len(unprojected_quads)*4, 2))
	tree = scipy.spatial.KDTree(unprojected_pts_with_outliers)
	dedup_dist = numpy.linalg.norm([.1, .1])
	dup_pairs = tree.query_pairs(dedup_dist)
	unique_filter = [True for _ in unprojected_pts_with_outliers]
	for pair in dup_pairs:
		sp = sorted(pair)
		dup_idx = sp[0]
		unique_filter[dup_idx] = False


	rotated_quads = [rotate_quad(quad) for quad in filtered_quads]
	scaled_quads = ([(x / avg_x_side, y / avg_y_side) for (x, y) in quad] for quad in rotated_quads)
	scaled_quads = list(scaled_quads)
	# All four corners get wound around to the same single corner
	wound_quads = ([
		(quad[0][0], quad[0][1]),
		(quad[1][0], quad[1][1] - 1),
		(quad[2][0] - 1, quad[2][1] - 1),
		(quad[3][0] - 1, quad[3][1]),
	] for quad in scaled_quads)
	# Translate each of the light and dark squares into the same space
	focused_quads = [[(x - int(is_light), y) for (x, y) in quad] for (quad, is_light) in zip(wound_quads, is_light_filtered_quads)]
	# Depending on the color of the square and the orientation of the corner,
	# the point either belongs on a NW-SE or a NE-SW diagonal corner.
	# The coordinates need to be rotated 45 degrees to take advantage of that symmetry.
	# FIXME: Use a more elegant calculation to rotate by 45 degrees
	x_values = (math.cos(math.atan2(y, x) + math.pi/4) * math.sqrt(x**2 + y**2) for quad in focused_quads for (x, y) in quad)
	y_values = (math.sin(math.atan2(y, x) + math.pi/4) * math.sqrt(x**2 + y**2) for quad in focused_quads for (x, y) in quad)
	# Plot the points on a unit circle, where it wraps around again at the boundaries of every 1-square-unit-area diamond
	x_radians = [x * 2*math.pi / math.sqrt(2) for x in x_values]
	y_radians = [y * 2*math.pi / math.sqrt(2) for y in y_values]
	# Now average all the vectors on the unit circle to find an average that is robust against wrapping
	offset_angle_x = math.atan2(
		sum(math.sin(angle) for angle in x_radians),
		sum(math.cos(angle) for angle in x_radians))
	offset_angle_y = math.atan2(
		sum(math.sin(angle) for angle in y_radians),
		sum(math.cos(angle) for angle in y_radians))
	# Use that to get the average offset
	avg_x_offset_rot = offset_angle_x / (2*math.pi) * math.sqrt(2)
	avg_y_offset_rot = offset_angle_y / (2*math.pi) * math.sqrt(2)
	# Rotate the offsets back off the 45 degree orientation
	avg_x_offset = math.cos(math.atan2(avg_y_offset_rot, avg_x_offset_rot) - math.pi/4) * math.sqrt(avg_x_offset_rot**2 + avg_y_offset_rot**2)
	avg_y_offset = math.sin(math.atan2(avg_y_offset_rot, avg_x_offset_rot) - math.pi/4) * math.sqrt(avg_x_offset_rot**2 + avg_y_offset_rot**2)
	# Shift all the quads so they should now lie on an integral grid
	transformed_quads = [[(x / avg_x_side - avg_x_offset, y / avg_y_side - avg_y_offset) for (x, y) in quad] for quad in filtered_quads]
	#print("TRANSFORMED", transformed_quads)
	# Snap all points to the nearest grid position
	snapped_quads = [[(int(round(x)), int(round(y))) for (x, y) in quad] for quad in transformed_quads]
	all_snapped_pts = [p for quad in snapped_quads for p in quad]
	all_transformed_pts = (p for quad in transformed_quads for p in quad)
	snap_distance = list(numpy.linalg.norm([sp[0] - tp[0], sp[1] - tp[1]]) for (sp, tp) in zip(all_snapped_pts, all_transformed_pts))
	outlier_snap_filter = identify_outliers(numpy.array(snap_distance))
	snapped_pts = numpy.array(all_snapped_pts)[outlier_snap_filter]
	transformed_pts = numpy.array([p for quad in transformed_quads for p in quad])[outlier_snap_filter]
	inlier_count = len(snapped_pts)
	# Calculate whether the light and dark squares are in the correct corners
	outlier_snap_quad_filter = [all(p) for p in grouper(outlier_snap_filter, 4)]
	is_light_snapped_quads = numpy.array(is_light_filtered_quads)[outlier_snap_quad_filter]
	if not len(is_light_snapped_quads):
		raise RuntimeError('No unambiguously-colored corners remain')
	kept_snapped_quads = numpy.array(snapped_quads)[outlier_snap_quad_filter]
	# The quad is on a light square if the first segment is vertical from the upper-left corner of a light square,
	# or horizontal from the upper-left corner of a dark square
	is_offset_all_snapped_quads = (bool((
		# Check whether the first corner is offset from an even grid point
		quad[0][0] + quad[0][1] +
		# Check whether the quad is known to be a light square
		int(is_light) +
		# Check whether the first segment is offset from a vertical side
		quad[1][0] - quad[0][0]
		) % 2)
		for (quad, is_light) in zip(kept_snapped_quads, is_light_snapped_quads))
	# All the values in is_offset should be the same
	is_offset_snapped = next(is_offset_all_snapped_quads)

	# Match the snapped points with their original image locations
	unprojected_pts = unprojected_pts_with_outliers[outlier_snap_filter]
	snapped_pts_coord = numpy.array([[float(x), float(y), 0.] for (x, y) in snapped_pts])

	#print('sorted snapped', sorted([tuple(p) for p in snapped_pts]))
	#print('snapped', [[tuple(p) for p in quad] for quad in snapped_quads])
	#print('unprojected', unprojected_pts)

	(h, w, _) = color3.shape
	f = max(h, w)
	fx = fy = f
	default_mtx = numpy.array([[fx, 0, (w - 1)/2.], [0, fy, (h - 1)/2.], [0, 0, 1]]).astype('float32')

	#print('before', r, tvec)
	newr, newj = cv2.Rodrigues(rvecs)
	#print('pose', rvecs, tvecs)
	#print('pose rodrigues', newr, tvecs)
	#print('homography', numpy.concatenate([newr, tvecs], axis=1))
	#cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv2.decomposeProjectionMatrix(numpy.concatenate([newr, tvecs], axis=1))
	#print('decomposed', cameraMatrix, rotMatrix, transVect)


	projm = numpy.concatenate([newr, tvecs], axis=1)
	homog = numpy.concatenate([numpy.delete(newr, 2, 1), tvecs], axis=1)
	r = homog
	#r = numpy.linalg.inv(homog)
	#tvec = r[:,2:]
	#print('trying', homog)



	min_snapped_x = min(x for (x, y) in snapped_pts) - 8
	min_snapped_y = min(y for (x, y) in snapped_pts) - 8
	#homog_src_pts = numpy.array([(float((x-min_snapped_x-7)*100), float((y-min_snapped_y-7)*100)) for (x, y) in snapped_pts])
	homog_src_pts = numpy.array([(float(x), float(y)) for (x, y) in snapped_pts])
	#print('pnp r', r)
	#print('snapped', homog_src_pts)
	#print('unprojected', unprojected_pts)
	fhomog, mask = cv2.findHomography(homog_src_pts, unprojected_pts)
	homog = numpy.dot(numpy.linalg.inv(default_mtx), fhomog)
	r = homog
	#print('homog r', r)
	#print('round trip', numpy.linalg.inv(fhomog), numpy.linalg.inv(numpy.dot(default_mtx, warp_h)))


	avg_x_side = 1.
	avg_y_side = 1.


	# Extend the grid to include all squares that could be part of the actual board
	min_snapped_x = min(x for (x, y) in snapped_pts) - 7
	min_snapped_y = min(y for (x, y) in snapped_pts) - 7
	max_snapped_x = max(x for (x, y) in snapped_pts) + 7
	max_snapped_y = max(y for (x, y) in snapped_pts) + 7
	is_offset_grid = bool((min_snapped_x + min_snapped_y + int(is_offset_snapped)) % 2)
	#grid_corners = list(itertools.product(
	#	range(int(min_snapped_x), int(max_snapped_x) + 1),
	#	range(int(min_snapped_y), int(max_snapped_y) + 1)))
	grid_corners = [[(x, y)
		for x in range(min_snapped_x, max_snapped_x + 1)]
		for y in range(min_snapped_y, max_snapped_y + 1)]
	grid_corners_coord = numpy.array([[float(x), float(y), 0.] for row_corners in grid_corners for (x, y) in row_corners])
	#project_grid_points_result, j = cv2.projectPoints(grid_corners_coord, rvecs, tvecs, default_mtx, dist)
	project_grid_points = cv2.perspectiveTransform(numpy.float32(grid_corners), fhomog)
	#print('GRID', min_snapped_x, max_snapped_x, min_snapped_y, max_snapped_y)
	project_grid_points_flat = project_grid_points.reshape(project_grid_points.shape[0] * project_grid_points.shape[1], 2)
	#project_grid_points = project_grid_points_result.reshape(max_snapped_y + 1 - min_snapped_y, max_snapped_x + 1 - min_snapped_x, 2)


	pimg = numpy.copy(color3)
	for quad in accurate_quads:
		cv2.drawContours(pimg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
	snapped_pts_coord = numpy.array([[x, y, 0.] for (x, y) in snapped_pts])
	#project_snapped_points, j = cv2.projectPoints(snapped_pts_coord, rvecs, tvecs, default_mtx, dist)
	project_snapped_points = cv2.perspectiveTransform(numpy.float32([snapped_pts]), fhomog)
	for cp in project_grid_points_flat:
		cv2.circle(pimg, (int(round(cp[0])), int(round(cp[1]))), 1, (0, 255, 0))
	for (idx, (cu, cp)) in enumerate(zip(unprojected_pts, project_snapped_points[0])):
		cv2.line(pimg, (int(round(cu[0])), int(round(cu[1]))), (int(round(cp[0])), int(round(cp[1]))), (255,0,0), 1)
		cv2.circle(pimg, (int(round(cu[0])), int(round(cu[1]))), 4, (0, 255, 255))
		cv2.circle(pimg, (int(round(cp[0])), int(round(cp[1]))), 4, (255, 0, 0))
	#cv2.imshow(WINNAME, pimg)
	#key = cv2.waitKey(0)


	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
	#print('points', project_grid_points)
	corners_flat = cv2.cornerSubPix(img1, numpy.float32(numpy.copy(project_grid_points_flat)), (5,5), (-1,-1), criteria)
	corners = corners_flat.reshape(*project_grid_points.shape)
	# TODO: Calculate the scores only for the relevant grid points
	# TODO: Also make sure the calculation has subpixel accuracy
	img1f = img1.astype('float')
	#corner_likelihood = cv2.log(cv2.Scharr(img1f, -1, 1, 0)**2 + cv2.Scharr(img1f, -1, 0, 1)**2)
	corner_likelihood = cv2.Scharr(img1f, -1, 1, 0)**2 + cv2.Scharr(img1f, -1, 0, 1)**2
	corner_orientation = cv2.phase(cv2.Scharr(img1f, -1, 1, 0), cv2.Scharr(img1f, -1, 0, 1))
	#corner_orientation = cv2.phase(cv2.Sobel(img1f, -1, 1, 0), cv2.Sobel(img1f, -1, 0, 1))
	#cv2.imshow(WINNAME, numpy.cos(corner_orientation)**2)
	#key = cv2.waitKey(0)

	pimg = numpy.copy(color3)
	for cp in project_grid_points_flat:
		cv2.circle(pimg, (int(round(cp[0])), int(round(cp[1]))), 3, (0, 255, 0))
	#for bp in corners_flat:
	#	cv2.circle(pimg, (int(round(bp[0])), int(round(bp[1]))), 3, (0, 255, 255))
	for bp in unprojected_pts:
		cv2.circle(pimg, (int(round(bp[0])), int(round(bp[1]))), 5, (255, 0, 0))
	#cv2.imshow(WINNAME, pimg)
	#key = cv2.waitKey(0)

	default_corner_likelihood = 0
	corner_scores = [[corner_likelihood.item(y, x)
		if x >= 0 and y >= 0 and x < corner_likelihood.shape[1] and y < corner_likelihood.shape[0]
		else default_corner_likelihood
			for (x, y) in ((int(round(cp[0])), int(round(cp[1]))) for cp in corners_row)]
			for corners_row in corners]
	debugimg = numpy.array([[(cell / 10, cell / 10, cell / 10) for cell in row] for row in corner_orientation])
	edge_scores = [[get_edge_likelihood((x, y), corners, corner_orientation) for x in range(corners.shape[1] * 2 - 1)] for y in range(corners.shape[0] * 2 - 1)]
	#print('edges', numpy.array_str(numpy.array(edge_scores)))
	max_edge_scores = rolling_sum(edge_scores, 17)
	#print('max', numpy.array_str(max_edge_scores))
	aligned_max_edge_scores = max_edge_scores[::2,::2]
	board_max_flat_idx = aligned_max_edge_scores.argmax()
	board_max_idx = numpy.unravel_index(board_max_flat_idx, aligned_max_edge_scores.shape[:2])
	is_offset_board = bool((board_max_idx[0] + board_max_idx[1] + int(is_offset_grid)) % 2)
	top_score = aligned_max_edge_scores.item(board_max_idx)
	contenders = list(itertools.takewhile(lambda x: x >= top_score*0.98, reversed(sorted([cell for row in aligned_max_edge_scores for cell in row]))))
	if len(contenders) > 1:
		print('top board positions', (contenders[0] - contenders[1]) / contenders[0] * 144 / 17, contenders)
	else:
		print('only top board position', contenders)
	best_corners = corners[board_max_idx[0]:board_max_idx[0]+9, board_max_idx[1]:board_max_idx[1]+9]
	#cv2.imshow(WINNAME, debugimg)
	#key = cv2.waitKey(0)

	#pimg = numpy.array([[tuple(reversed(colorsys.hls_to_rgb(cell, 0.5, 1.))) for cell in row] for row in corner_orientation])
	pimg = numpy.copy(debugimg)
	for y in range(project_grid_points.shape[0]):
		for x in range(project_grid_points.shape[1]):
			cp = project_grid_points[y][x]
			grid_color = (0, 1., 0)
			try:
				score = aligned_max_edge_scores[y][x]
				if score >= top_score*0.98:
					grid_color = (0, 0, 1.)
			except IndexError:
				pass
			cv2.circle(pimg, (int(round(cp[0])), int(round(cp[1]))), 4, grid_color)
	for bp in best_corners.reshape(best_corners.shape[0] * best_corners.shape[1], best_corners.shape[2]):
		cv2.circle(pimg, (int(round(bp[0])), int(round(bp[1]))), 4, (0, 255, 255))
	for y in range(project_grid_points.shape[0]):
		for x in range(project_grid_points.shape[1]):
			cp = project_grid_points[y][x]
			try:
				score = aligned_max_edge_scores[y][x]
				if score >= top_score*0.98:
					grid_color = (0, 0, 1.)
				else:
					continue
			except IndexError:
				continue
			cv2.circle(pimg, (int(round(cp[0])), int(round(cp[1]))), 4, grid_color)
	#for bp in best_corners.reshape(best_corners.shape[0] * best_corners.shape[1], best_corners.shape[2]):
	#	cv2.circle(corner_likelihood, (int(round(bp[0])), int(round(bp[1]))), 8, 255)
	#cv2.imshow(WINNAME, pimg)
	#cv2.imshow(WINNAME, corner_likelihood*10000000)
	#key = cv2.waitKey(0)


	if is_offset_board:
		best_corners = numpy.array([[best_corners[y][x] for y in range(9)] for x in range(8, -1, -1)])

	return best_corners


def get_projection(corners, shape):
	(h, w, _) = shape
	f = max(h, w)
	fx = fy = f
	default_mtx = numpy.array([[fx, 0, (w - 1)/2.], [0, fy, (h - 1)/2.], [0, 0, 1]]).astype('float32')

	best_corners_input = corners.reshape(1, 81, 2).astype('float32')
	rotated_grid = numpy.float32([[[float(x), float(y), 0.] for x in range(9) for y in range(9)]])
	err, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
		rotated_grid,
		best_corners_input,
		(shape[1], shape[0]),
		default_mtx,
		numpy.zeros(5).astype('float32'),
		flags=(
			cv2.CALIB_USE_INTRINSIC_GUESS +
			cv2.CALIB_FIX_PRINCIPAL_POINT +
			cv2.CALIB_FIX_ASPECT_RATIO +
			cv2.CALIB_ZERO_TANGENT_DIST +
			cv2.CALIB_FIX_K1 +
			cv2.CALIB_FIX_K2 +
			cv2.CALIB_FIX_K3
		)
	)
	return pose.Projection(pose.CameraIntrinsics(cameraMatrix=cameraMatrix, distCoeffs=distCoeffs), pose.Pose(rvec=rvecs[0], tvec=tvecs[0]))


def get_edge_likelihood(point, corners, corner_orientation):
	"""Return the strength of a candidate edge according to the image of edges

	The resulting matrix of points forms a matrix where every other element is zero:
	0 - 0 - 0
        | 0 | 0 |
	0 - 0 - 0
        | 0 | 0 |
	0 - 0 - 0
	Even rows contain horizontal edge scores and zero'd corners.
	Odd ros contain vertical edge scores and zero'd centers.
	"""
	(grid_x, grid_y) = point
	if not (grid_x + grid_y) % 2:
		return 0
	if grid_x % 2:
		# Calculate the horizontal edge
		(corner_x1_idx, corner_y1_idx) = (grid_x // 2, grid_y // 2)
		(corner_x2_idx, corner_y2_idx) = (corner_x1_idx + 1, corner_y1_idx)
	else:
		# Calculate the vertical edge
		(corner_x1_idx, corner_y1_idx) = (grid_x // 2, grid_y // 2)
		(corner_x2_idx, corner_y2_idx) = (corner_x1_idx, corner_y1_idx + 1)
	(corner_x1, corner_y1) = corners[corner_y1_idx][corner_x1_idx]
	(corner_x2, corner_y2) = corners[corner_y2_idx][corner_x2_idx]
	target_angle = math.atan2(corner_y2 - corner_y1, corner_x2 - corner_x1) + math.pi/2
	total_score = 0
	total_weight = 0
	for ((x, y), weight) in draw_line_iter((corner_x1, corner_y1), (corner_x2, corner_y2)):
		if x < 0 or y < 0 or x >= corner_orientation.shape[1] or y >= corner_orientation.shape[0]:
			continue
		item = corner_orientation.item(y, x)
		total_weight += weight
		match = math.cos(item - target_angle)**2
		score = match * weight
		total_score += score
	score = (total_score / total_weight) if total_weight else 0.
	return score


def _fpart(x):
    return x - int(x)


def _rfpart(x):
    return 1 - _fpart(x)


def draw_line_iter(p1, p2):
    """Draws an anti-aliased line from p1 to p2

    Adapted from https://rosettacode.org/wiki/Xiaolin_Wu%27s_line_algorithm#Python"""
    x1, y1, x2, y2 = p1 + p2
    dx, dy = x2-x1, y2-y1
    steep = abs(dx) < abs(dy)
    p = lambda px, py: ((px,py), (py,px))[steep]

    if steep:
        x1, y1, x2, y2, dx, dy = y1, x1, y2, x2, dy, dx
    if x2 < x1:
        x1, x2, y1, y2 = x2, x1, y2, y1

    grad = dy/float(dx)
    intery = y1 + _rfpart(x1) * grad
    def draw_endpoint(pt):
        x, y = pt
        xend = round(x)
        yend = y + grad * (xend - x)
        xgap = _rfpart(x + 0.5)
        px, py = int(xend), int(yend)
        return (px, [((px, py), _rfpart(yend) * xgap), ((px, py+1), _fpart(yend) * xgap)])

    (xstart, pixels) = draw_endpoint(p(*p1))
    for pixel in pixels:
        yield pixel
    (xend, pixels) = draw_endpoint(p(*p2))
    for pixel in pixels:
        yield pixel
    if xstart > xend:
        (xstart, xend) = (xend, xstart)

    for x in range(xstart + 1, xend):
        y = int(intery)
        yield (p(x, y), _rfpart(intery))
        yield (p(x, y+1), _fpart(intery))
        intery += grad


def filter_quads(contours):
	approxes = []
	for contour in contours:
		if len(contour) > 4:
			perimeter = cv2.arcLength(contour, closed=True)
			approx = cv2.approxPolyDP(contour, perimeter/20, True)
			#if len(approx) > 4:
			#	approx = cv2.approxPolyDP(approx, 5, True)
			if len(approx) != 4:
				continue
		elif len(contour) == 4:
			approx = contour
		else:
			continue
		approx = numpy.squeeze(approx)

		# Any negative-oriented (clockwise) contours are rejected.
		if numpy.cross(approx[1] - approx[0], approx[2] - approx[1]) >= 0:
			continue
		if not cv2.isContourConvex(approx):
			continue
		if cv2.contourArea(approx) < 40:
			continue

		approxes.append(approx)
	return approxes


def get_dilated_kernel_hull(kernel):
	kernel_cloud = [(kpx - kernel.shape[0]//2, kpy - kernel.shape[1]//2)
		for kpx in range(kernel.shape[0]) for kpy in range(kernel.shape[1])
		if kernel[(kpx,kpy)]]
	kernel_hull = [p[0] for p in cv2.convexHull(numpy.array(kernel_cloud))]
	return kernel_hull


def dilate_contours(contours, kernel_hull):
	filtered = []

	for approx in contours:
		dilated_segments = []
		# FIXME: This loop is even slower than running drawContours and findContours below
		for segment_idx in range(len(approx)):
			#print('-----------------')
			#print('kernel', kernel_hull)
			segment = numpy.array([approx[segment_idx], approx[(segment_idx + 1) % len(approx)]])
			segment_vector = segment[1] - segment[0]
			segment_unit = segment_vector / numpy.linalg.norm(segment_vector)

			#print('segment', segment_vector)
			rejection_vectors = (numpy.dot(kp, segment_unit) * segment_unit - kp
				for kp in kernel_hull)
			#print('segment', segment_vector, segment_unit)
			#print('rejections', zip(rejection_vectors, kernel_hull))
			offset = max(rejection_vectors, key=lambda rv: numpy.cross(segment_vector, rv))
			#print('offset', offset)
			dilated_segment = [p + offset for p in segment]
			#print('dilated_segment', dilated_segment)
			dilated_segments.append(dilated_segment)

		dilated_contour = []
		for segment_idx in range(len(dilated_segments)):
			segment = dilated_segments[segment_idx]
			next_segment = dilated_segments[(segment_idx + 1) % len(dilated_segments)]
			intersection = line_intersection(segment, next_segment)
			#print('intersection', intersection)
			dilated_contour.append(intersection)
		#print('contour', dilated_contour)
		#dilated_contour_array = numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1)))
		#	for (x,y) in dilated_contour])
		dilated_contour_array = numpy.array(dilated_contour)

		filtered.append(dilated_contour_array)
	return filtered


def is_complementary_corner(left_idx, right_idx, quads, color2):
	"""Tests whether the segments emanating from two quads' shared corner are collinear"""
	left_quad = quads[left_idx//4]
	right_quad = quads[right_idx//4]
	left_corner_idx = left_idx % 4
	right_corner_idx = right_idx % 4
	left_segments = [[left_quad[left_corner_idx], left_quad[(left_corner_idx + 1) % 4]],
		[left_quad[left_corner_idx], left_quad[(left_corner_idx + 3) % 4]]]
	right_segments = [[right_quad[right_corner_idx], right_quad[(right_corner_idx + 1) % 4]],
		[right_quad[right_corner_idx], right_quad[(right_corner_idx + 3) % 4]]]
	scores = [sorted([get_collinear_score(left_segments[0], right_segments[0]), get_collinear_score(left_segments[1], right_segments[1])]),
		sorted([get_collinear_score(left_segments[0], right_segments[1]), get_collinear_score(left_segments[1], right_segments[0])])]
	# We want quads where the scores are [-1, -1] or else [-1, 1]
	# We make twisted scores so that the best scores are [-1, -1]
	twisted_scores = [[pair[0], -abs(pair[1])] for pair in scores]
	combined_score = min(sum((s + 1)**2 for s in pair) for pair in twisted_scores)
	success = combined_score < 0.002
	#if success:
	#	print('Collinear', scores)
	#else:
	#	print('NOT', scores)

	#contlines = numpy.zeros((color2.shape[0], color2.shape[1], 3), numpy.uint8)
	#for contour in [left_quad, right_quad]:
	#	cv2.drawContours(contlines, numpy.array([[(int(numpy.clip(x, 0, contlines.shape[1]-1)), int(numpy.clip(y, 0, contlines.shape[0]-1)))
	#	for (x,y) in contour]]), -1, (255, 0, 0), 1)
	#cv2.imshow(WINNAME, contlines)
	#key = cv2.waitKey(0)

	return success


def get_collinear_score(left_segment, right_segment):
	"""Test collinearity. Success results in 1 or -1 return value"""
	left_vector = left_segment[1] - left_segment[0]
	right_vector = right_segment[1] - right_segment[0]
	dot = numpy.dot(left_vector, right_vector)
	# The best normalized score is -1.
	normalized = dot / (numpy.linalg.norm(left_vector) * numpy.linalg.norm(right_vector))
	return normalized


def rolling_sum(a, n=9):
	na = numpy.array(a)
	return scipy.signal.convolve2d(na, numpy.ones((n, n), dtype=float), 'valid')


def euclidean_distance (p1, p2):
	"""
		Function: euclidean_distance 
		----------------------------
		given two points as 2-tuples, returns euclidean distance 
		between them
	"""
	assert ((len(p1) == len(p2)) and (len(p1) == 2))
	return numpy.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_centroid (points):
	"""
		Function: get_centroid 
		----------------------
		given a list of points represented 
		as 2-tuples, returns their centroid 
	"""
	#return (numpy.mean([p[0] for p in points]), numpy.mean([p[1] for p in points]))
	c = shapely.geometry.polygon.Polygon(points).centroid
	return (c.x, c.y)


def angle_between(v1, v2):
	v1_norm = numpy.linalg.norm(v1)
	v2_norm = numpy.linalg.norm(v2)
	if v1_norm == 0. or v2_norm == 0:
		return 0.
	angle = numpy.arccos(numpy.clip(numpy.dot(v1 / v1_norm, v2 / v2_norm), -1., 1.))
	# Shift the range from [0, pi) to [-pi/2, pi/2)
	return (angle + math.pi/2.) % math.pi - math.pi/2.


def line_intersection(a, b):
	((a1, a2), (b1, b2)) = (a, b)
	da = a2 - a1
	db = b2 - b1
	dp = a1 - b1
	dap = (-da[1], da[0])
	denom = numpy.dot(dap, db)
	num = numpy.dot(dap, dp)
	return (num / denom.astype(float))*db + b1


def get_perimeter(contour):
	(left, right, wrap) = itertools.tee(contour, 3)
	next(right, None)
	pairs = zip(left, itertools.chain(right, wrap))
	return sum(numpy.linalg.norm(end - start) for (start, end) in pairs)


def get_best_intersection_by_dist(segments):
	dims = segments.shape[1]
	vectors = (end - begin for (begin, end) in segments)
	measured_vectors = ((vector, numpy.linalg.norm(vector)) for vector in vectors)
	unit_vectors = [vector / mag if mag != 0. else numpy.zeros(dims)
		for (vector, mag) in measured_vectors]
	diagonals = numpy.eye(dims)
	sums = numpy.array([[
		sum(vector[dimx] * vector[dimy] - diagonals[dimy][dimx] for vector in unit_vectors)
		for dimx in range(dims)]
		for dimy in range(dims)])
	begins = numpy.array([
		sum(segment[0][dimx] * (vector[dimx] * vector[dimy] - diagonals[dimy][dimx])
		for dimx in range(dims) for (segment, vector) in zip(segments, unit_vectors))
		for dimy in range(dims)])
	#print('sums', sums)
	#print('begins', begins)
	result = numpy.linalg.lstsq(sums, begins)
	#print('lstsq result', result)
	intersect = result[0].reshape(dims)
	return intersect


def get_best_intersection_by_angle1(segments, precalculated=None):
	""" Solve for d = 1 - (x/|x|) . n and
	df/dp = 2 * (1 - (x/|x|) . n) * (I * |x| - x * (x^T/|x|)) . n, x = a - p"""
	print('******************** get_best_intersection_by_angle1')
	dims = segments.shape[1]
	print('SEGMENTS', segments)
	vectors = [end - begin for (begin, end) in segments]
	print('VECTORS', vectors)
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	print('VECTOR_DISTS', vector_dists)
	# n ~ unit_vectors
	unit_vectors = [vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists)]
	print('UNIT_VECTORS_INPUT', unit_vectors)
	pivots = [numpy.mean(segment, axis=0) for segment in segments]
	# FIXME: Should this be numpy.eye(dims) ?
	one = numpy.ones(dims)

	seed = numpy.mean(pivots, axis=0)
	context = (unit_vectors, pivots)
	def objective(vp, unit_vectors, pivots):
		print('--- OBJECTIVE')
		print('vp', repr(vp))
		print('unit_vectors', repr(unit_vectors))
		print('pivots', repr(pivots))
		reaches = [vp - pivot for pivot in pivots]
		dists = (numpy.linalg.norm(reach) for reach in reaches)
		unit_reaches = (reach/dist if dist != 0. else numpy.zeros(reach.shape)
			for (reach, dist) in zip(reaches, dists))
		unit_reaches = list(unit_reaches)
		print('unit_reaches', repr(unit_reaches))
		# FIXME: Because of the "abs", there are local minima!
		residuals = [1 - abs(numpy.dot(unit_reach, unit_vector))
			for (unit_vector, unit_reach) in zip(unit_vectors, unit_reaches)]
		print('residuals', residuals)
		print('sum-squares', sum(residual**2 for residual in residuals))
		return residuals
	def jacobian(vp, unit_vectors, pivots):
		reaches = [vp - pivot for pivot in pivots]
		dists = [numpy.linalg.norm([pivot, vp]) for reach in reaches]
		unit_reaches = (reach/dist if dist != 0. else numpy.zeros(reach.shape)
			for (reach, dist) in zip(reaches, dists))
		outers = (numpy.outer(reach, reach) / dist for (reach, dist) in zip(reaches, dists))
		return [2 * (1 - numpy.dot(unit_reach, unit_vector)) *
			numpy.dot(one * dist - outer, unit_vector)
			for (unit_vector, unit_reach, dist, outer) in
				zip(unit_vectors, unit_reaches, dists, outers)]
	# FIXME
	jacobian = None
	try:
		result = scipy.optimize.leastsq(objective, seed, context, jacobian)
		print('leastsq result', result)
		if precalculated is not None:
			print('precalculated vp', precalculated)
			print('precalculated objective', objective(precalculated, unit_vectors, pivots))
		intersect = result[0]
		return intersect
	except:
		print('exception', sys.exc_info()[0], traceback.format_exc())


def get_distance(S, p, C):
	d = numpy.dot(S, p) - C
	return (sum(e**2 for e in d), d)


def get_best_intersection_by_angle2(segments, precalculated=None):
	""" Solve for MIN [ D = (b - a) . (x - a) ] where (b-a) is perpendicular to the segment,
	or d/dx D^2 = (b - a) . x - (b - a) . a = 0
	"""
	print('******************** get_best_intersection_by_angle2')
	dims = segments.shape[1]
	vectors = (end - begin for (begin, end) in segments)
	pivots = (numpy.mean(segment, axis=0) for segment in segments)
	measured_vectors = ((vector, numpy.linalg.norm(vector)) for vector in vectors)
	unit_vectors = (vector / mag if mag != 0. else numpy.zeros(dims)
		for (vector, mag) in measured_vectors)
	perpendicular_unit_vectors = [numpy.array([vector[1], -vector[0]])
		for vector in unit_vectors]
	products = [numpy.dot(vector, pivot)
		for (vector, pivot) in zip(perpendicular_unit_vectors, pivots)]
	print('perpendicular_unit_vectors', perpendicular_unit_vectors)
	print('products', products)
	result = numpy.linalg.lstsq(perpendicular_unit_vectors, products)
	print('lstsq result', result)
	intersect = result[0]
	print('intersect', intersect)
	print('analytical distances', get_distance(perpendicular_unit_vectors, intersect, products))
	print('numerical distances', get_distance(perpendicular_unit_vectors, [1000., 548.87250946], products))
	return intersect


def get_best_intersection_by_angle3(segments, precalculated=None):
	""" Solve for MIN [ D = ((b - a) . (x - a)) / ||x - a|| ]
	or d/dx D^2 = ((b - a) . (x - a)) * (||x - a|| - ((b - a) . (x - a))) = 0"""
	print('******************** get_best_intersection_by_angle3')
	dims = segments.shape[1]
	print('SEGMENTS', segments)
	vectors = [end - begin for (begin, end) in segments]
	print('VECTORS', vectors)
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	print('VECTOR_DISTS', vector_dists)
	unit_vectors = (vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists))
	unit_vectors = list(unit_vectors)
	# pivots ~ a
	pivots = [numpy.mean(segment, axis=0) for segment in segments]
	# perpendicular_unit_vectors ~ b - a
	perpendicular_unit_vectors = [numpy.array([vector[1], -vector[0]])
		for vector in unit_vectors]
	# products ~ (b - a) . a
	products = [numpy.dot(vector, pivot)
		for (vector, pivot) in zip(perpendicular_unit_vectors, pivots)]
	print('PERPENDICULAR_UNIT_VECTORS_INPUT', perpendicular_unit_vectors)

	seed = numpy.mean(pivots, axis=0)
	#seed[0] = seed[0] + 1000
	context = (perpendicular_unit_vectors, pivots)
	def objective(vp, unit_vectors, pivots):
		#print('--- OBJECTIVE')
		#print('vp', repr(vp))
		#print('perpendicular_unit_vectors', repr(perpendicular_unit_vectors))
		#print('pivots', repr(pivots))
		# reaches ~ x - a
		reaches = [vp - pivot for pivot in pivots]
		reach_dists = (numpy.linalg.norm(reach) for reach in reaches)
		unit_reaches = (reach/reach_dist if reach_dist != 0. else numpy.zeros(reach.shape)
			for (reach, reach_dist) in zip(reaches, reach_dists))
		unit_reaches = list(unit_reaches)
		#print('unit_reaches', repr(unit_reaches))
		projections = (numpy.dot(perpendicular_unit_vector, reach)
			for (perpendicular_unit_vector, reach) in
			#zip(perpendicular_unit_vectors, reaches))
			zip(unit_vectors, reaches))
		projections = list(projections)
		#print('projections', projections)
		residuals = [math.asin(projection / numpy.linalg.norm(reach)) * 180. / math.pi
			for (projection, reach) in zip(projections, reaches)]
		#print('residuals', residuals)
		#print('sum-squares', sum(residual**2 for residual in residuals))
		return residuals
	# FIXME
	jacobian = None
	try:
		result = scipy.optimize.leastsq(objective, seed, context, jacobian)
		print('leastsq result', result)
		if precalculated is not None:
			print('precalculated vp', precalculated)
			print('precalculated objective', objective(precalculated, unit_vectors, pivots))
		intersect = result[0]
		return intersect
	except:
		print('exception', sys.exc_info()[0], traceback.format_exc())



def get_best_intersection_by_angle4(segments, precalculated=None):
	print('******************** get_best_intersection_by_angle4')
	dims = segments.shape[1]
	print('SEGMENTS', segments)
	vectors = [end - begin for (begin, end) in segments]
	print('VECTORS', vectors)
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	print('VECTOR_DISTS', vector_dists)
	unit_vectors = (vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists))
	unit_vectors = list(unit_vectors)
	# pivots ~ a
	pivots = [numpy.mean(segment, axis=0) for segment in segments]
	# perpendicular_unit_vectors ~ b - a
	perpendicular_unit_vectors = [numpy.array([vector[1], -vector[0]])
		for vector in unit_vectors]
	# products ~ (b - a) . a
	products = [numpy.dot(vector, pivot)
		for (vector, pivot) in zip(perpendicular_unit_vectors, pivots)]
	print('PERPENDICULAR_UNIT_VECTORS_INPUT', perpendicular_unit_vectors)

	seed = numpy.mean(pivots, axis=0)
	#seed[0] = seed[0] + 1000
	context = (perpendicular_unit_vectors, pivots)
	def objective(vp, unit_vectors, pivots):
		print('--- OBJECTIVE')
		print('vp', repr(vp))
		print('perpendicular_unit_vectors', repr(perpendicular_unit_vectors))
		print('pivots', repr(pivots))
		# reaches ~ x - a
		reaches = [vp - pivot for pivot in pivots]
		residuals = [angle_between(vector, reach)
			for (vector, reach) in zip(vectors, reaches)]
		print('residuals', residuals)
		print('sum-squares', sum(residual**2 for residual in residuals))
		return residuals
	# FIXME
	jacobian = None
	try:
		result = scipy.optimize.leastsq(objective, seed, context, jacobian)
		print('leastsq result', result)
		if precalculated is not None:
			print('precalculated vp', precalculated)
			print('precalculated objective', objective(precalculated, unit_vectors, pivots))
		intersect = result[0]
		return intersect
	except:
		print('exception', sys.exc_info()[0], traceback.format_exc())


def get_best_intersection_by_angle5(segments, precalculated=None):
	#print('******************** get_best_intersection_by_angle5')
	#print('SEGMENTS', segments)
	#return get_best_intersection_by_angle5_inner(segments)
	segments_flipped = ([[segment[1], segment[0]] if flip else segment
		for (segment, flip) in zip(segments, orientation)]
		for orientation in itertools.product((False, True), repeat=len(segments)))
	costs = (get_best_intersection_by_angle5_inner(numpy.array(segments)) for segments in segments_flipped)
	best = min(costs, key=lambda cost: cost[0])
	#print('BEST', best)
	return best[1]


def get_error_by_angle5_quad(vps, quad):
	quads_rotated = [quad, quad[1:] + quad[:1]]
	costs = (get_error_by_angle5_rotated_quad(vps, quad)
		for quad in quads_rotated)
	best = min(costs)
	#print('get_error_by_angle5_quad', best)
	return best

def get_max_error_by_angle5_quad(vps, quad):
	quads_rotated = [quad, quad[1:] + quad[:1]]
	costs = (get_max_error_by_angle5_rotated_quad(vps, quad)
		for quad in quads_rotated)
	best = min(costs)
	#print('get_max_error_by_angle5_quad', best)
	return best

hunts = 0
hunts_inner = 0
def get_best_intersection_by_angle5_quads_exponential(quads):
	global hunts
	hunts = 0
	#print('******************** get_best_intersection_by_angle5_quads_exponential')
	# The combination where all quads are rotated is symmetric.
	# So the first quad should never be rotated.
	rotations = (itertools.chain((False,), rotation)
		for rotation in itertools.product((False, True), repeat=len(quads) - 1))
	quads_rotated = ([quad[1:] + quad[:1] if rotate else quad
		for (quad, rotate) in zip(quads, rotation)] for rotation in rotations)
	quads_rotated = list(quads_rotated)
	costs = (get_best_intersection_by_angle5_rotated_quads(quads) for quads in quads_rotated)
	best = min(costs, key=lambda cost: cost[0])
	#print('COST', best[0])
	print('hunts', len(quads), hunts)
	return best[1]

def get_all_orientations(quad):
	# FIXME: Calculate rather than hard-code these indices
	return [
		# Unrotated
		(
			# VP 1
			[
				# Unflipped
				[[quad[0], quad[1]], [quad[3], quad[2]]],
				# Flipped
				[[quad[1], quad[0]], [quad[2], quad[3]]],
			],
			# VP 2
			[
				# Unflipped
				[[quad[1], quad[2]], [quad[0], quad[3]]],
				# Flipped
				[[quad[2], quad[1]], [quad[3], quad[0]]],
			],
		),
		# Rotated
		(
			# VP 1
			[
				# Unflipped
				[[quad[1], quad[2]], [quad[0], quad[3]]],
				# Flipped
				[[quad[2], quad[1]], [quad[3], quad[0]]],
			],
			# VP 2
			[
				# Unflipped
				[[quad[0], quad[1]], [quad[3], quad[2]]],
				# Flipped
				[[quad[1], quad[0]], [quad[2], quad[3]]],
			],
		),
	]

def add_best_quad_orientation(hypothesis, quad):
	orientations = get_all_orientations(quad)
	(vp1, vp2) = (hypothesis[0][1], hypothesis[1][1])
	results = (get_best_segment_orientation((vp1, vp2), orientation) for orientation in orientations)
	best = min(results, key=lambda e: e[0])

	# When adding segments to the hypothesis, don't update the cost, since it is inaccurate.
	# To get an accurate cost, the VP needs to be adjusted for the new segments.
	return ((hypothesis[0][0], vp1, hypothesis[0][2] + best[1]), (hypothesis[1][0], vp2, hypothesis[1][2] + best[2]))

def get_best_segment_orientation(vps, orientation):
	(vp1, vp2) = vps
	(vp1_attempts, vp2_attempts) = orientation

	segment_costs1 = ((get_error_by_angle5_segments(vp1, numpy.array(attempt)), attempt) for attempt in vp1_attempts)
	segment_costs2 = ((get_error_by_angle5_segments(vp2, numpy.array(attempt)), attempt) for attempt in vp2_attempts)

	best1 = min(segment_costs1, key=lambda e: e[0])
	best2 = min(segment_costs2, key=lambda e: e[0])

	return (best1[0] + best2[0], best1[1], best2[1])


def get_best_intersection_by_angle5_quads(quads, tol=math.pi/36000.):
	global hunts
	hunts = 0
	#print('******************** get_best_intersection_by_angle5_quads')

	# For each VP, track the cost, coordinates, and segments
	hypotheses = [((0, (0, 0), []), (0, (0, 0), []))]
	is_first = True
	for idx, quad in enumerate(quads):
		#print('hypotheses', len(hypotheses), '{}/{}'.format(idx, len(quads)))
		if len(hypotheses) > 10:
			working = numpy.copy(color_global)
			for hypothesis in hypotheses:
				#print('hypothesis err', hypothesis[0][0], hypothesis[1][0])
				working = numpy.copy(color_global)

				(vp1, vp2) = (hypothesis[0][1], hypothesis[1][1])
				for segment in hypothesis[0][2] + hypothesis[1][2]:
					cv2.line(working, tuple(int(p) for p in segment[0]), tuple(int(p) for p in segment[1]), (0,255,0), 2)
				if any(dim > 1e9 or dim < -1e9 for vp in (vp1, vp2) for dim in vp):
					#print('overflow', (vp1, vp2))
					continue

				cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
				cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
				cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 2)

				for segment in hypothesis[0][2]:
					for point in segment:
						cv2.line(working, tuple(int(dim) for dim in point), tuple(int(d) for d in vp1), (0,0,255), 1)
				for segment in hypothesis[1][2]:
					for point in segment:
						cv2.line(working, tuple(int(dim) for dim in point), tuple(int(d) for d in vp2), (0,0,255), 1)

				cv2.imshow(WINNAME, working)
				key = cv2.waitKey(1)

		orientations = get_all_orientations(quad)
		if is_first:
			orientations = itertools.islice(orientations, 1)
			# FIXME: Set vp seed in hypotheses to the average of all segments
			is_first = False
		# FIXME: Instead of assuming 10 iters, reorder the quads so the later ones are known not to move the VP much
		elif idx > 10 and idx < len(quads) - 1:
			is_shortcut = True
			# Deduce the best orientation and add segments to each hypothesis.
			hypotheses = [add_best_quad_orientation(hypothesis, quad) for hypothesis in hypotheses]
		else:
			#print('***comb', sum(len(orientation) for orientation in get_all_orientations(quad)), '*', len(hypotheses))
			attempts = ((
				# All the qualifying checks for VP1 will need to be
				# combined pairwise with all the qualifying checks for VP2.
				# VP1
				# FIXME: Use a loop comprehension to get all the flips.
				(
					hypothesis[0][1],
					[
						# Unflipped for VP 1
						hypothesis[0][2] + orientation[0][0],
						# Flipped for VP 1
						hypothesis[0][2] + orientation[0][1],
					],
				),
				# VP2
				(
					hypothesis[1][1],
					[
						# Unflipped for VP 2
						hypothesis[1][2] + orientation[1][0],
						# Flipped for VP 2
						hypothesis[1][2] + orientation[1][1],
					],
				),
			) for (hypothesis, orientation) in itertools.product(hypotheses, orientations))
			# Measure each attempt and filter out the ones that don't qualify.
			hypotheses = [qualifier for attempt in attempts for qualifier in measure_attempt(
				attempt[0][0], attempt[0][1], attempt[1][0], attempt[1][1], tol, quad)]
			if not hypotheses:
				# Nothing qualified
				print('None qualified', len(quads), hunts)
				return None

	best = min(hypotheses, key=lambda hypothesis: hypothesis[0][0] + hypothesis[1][0])

	#print('costs', [hypothesis[0][0] + hypothesis[1][0] for hypothesis in hypotheses])
	#print('COST', best[0][0] + best[1][0])
	#print('HUNTS', len(quads), hunts)
	return (best[0][1], best[1][1])

# FIXME
THRESHOLD = math.pi/120.
# Input: [VP1 unflipped list of segments, VP1 flipped list of segments], [VP2 unflipped list of segments, VP2 flipped list of segments]
# Output: list of each ((VP1 cost, VP1 coordinates, VP1 segments), (VP2 cost, VP2 coordinates, VP2 segments))
def measure_attempt(vp1, vp1_attempts, vp2, vp2_attempts, tol, last_quad):
	global hunts_inner
	hunts_inner = 0
	measurements1 = (measure_oriented_attempt(vp1, segments, tol) for segments in vp1_attempts)
	measurements2 = (measure_oriented_attempt(vp2, segments, tol) for segments in vp2_attempts)
	measurements1_filtered = (filtered[0] for filtered in measurements1 if all(segment_cost < THRESHOLD for segment_cost in filtered[1]))
	measurements2_filtered = (filtered[0] for filtered in measurements2 if all(segment_cost < THRESHOLD for segment_cost in filtered[1]))
	segment_count = len(vp1_attempts[1])
	measurements1_filtered = list(measurements1_filtered)
	measurements2_filtered = list(measurements2_filtered)
	#print('measure_attempt', segment_count, THRESHOLD, measurements1_filtered, '----', measurements2_filtered)
	#print('error', get_error_by_angle5_segments(measurements1_filtered[0][1], numpy.array(measurements1_filtered[0][2])), get_error_by_angle5_segments(measurements2_filtered[0][1], numpy.array(measurements2_filtered[0][2])))
	#print('retry', segment_count, tol/10000, [measure_oriented_attempt(segments, tol/10000) for segments in vp1_attempts], '----', [measure_oriented_attempt(segments, tol/10000) for segments in vp2_attempts])
	hypotheses = (qualifier
		for qualifier in itertools.product(measurements1_filtered, measurements2_filtered)
		if qualifier[0][0] + qualifier[1][0] < THRESHOLD * segment_count)
	hypotheses = list(hypotheses)
	#print('hunts_inner', hunts_inner)
	return hypotheses

# Input: list of segments
# Output: ((cost, coordinates, segments), segment-costs)
def measure_oriented_attempt(vp_seed, segments, tol):
	(cost, vp) = get_best_intersection_by_angle5_segments(vp_seed, numpy.array(segments), tol)
	#print('measure_oriented_attempt', cost, len(segments))
	segment_costs = (get_error_by_angle5_segments(vp, numpy.array([segment])) for segment in segments)
	return ((cost, vp, segments), segment_costs)

def get_error_by_angle5_rotated_quad(vps, quad):
	cost1 = get_error_by_angle5_vp(vps[0], quad)
	cost2 = get_error_by_angle5_vp(vps[1], quad[1:] + quad[:1])
	return cost1 + cost2

def get_max_error_by_angle5_rotated_quad(vps, quad):
	cost1 = get_max_error_by_angle5_vp(vps[0], quad)
	cost2 = get_max_error_by_angle5_vp(vps[1], quad[1:] + quad[:1])
	return max(cost1, cost2)

def get_best_intersection_by_angle5_rotated_quads(quads):
	(cost1, vp1) = get_best_intersection_by_angle5_vp(quads)
	(cost2, vp2) = get_best_intersection_by_angle5_vp([quad[1:] + quad[:1] for quad in quads])

	return (cost1 + cost2, (vp1, vp2))


def get_best_intersection_by_angle5_vp(quads):
	orientations = itertools.product((False, True), repeat=len(quads))
	segments_oriented = ([[[quad[1], quad[0]], [quad[2], quad[3]]] if flip else [[quad[0], quad[1]], [quad[3], quad[2]]]
		for (quad, flip) in zip(quads, orientation)]
		for orientation in orientations)
	segments_flattened = ([segment for pair in segments for segment in pair] for segments in segments_oriented)
	costs = (get_best_intersection_by_angle5_segments(None, numpy.array(segments))
		for segments in segments_flattened)
	best = min(costs, key=lambda cost: cost[0])
	return best

def get_error_by_angle5_vp(vp, quad):
	quads = [quad]
	orientations = itertools.product((False, True), repeat=len(quads))
	segments_oriented = ([[[quad[1], quad[0]], [quad[2], quad[3]]] if flip else [[quad[0], quad[1]], [quad[3], quad[2]]]
		for (quad, flip) in zip(quads, orientation)]
		for orientation in orientations)
	segments_flattened = ([segment for pair in segments for segment in pair] for segments in segments_oriented)
	costs = (get_error_by_angle5_segments(vp, numpy.array(segments))
		for segments in segments_flattened)
	best = min(costs)
	return best

def get_max_error_by_angle5_vp(vp, quad):
	quads = [quad]
	orientations = itertools.product((False, True), repeat=len(quads))
	segments_oriented = ([[[quad[1], quad[0]], [quad[2], quad[3]]] if flip else [[quad[0], quad[1]], [quad[3], quad[2]]]
		for (quad, flip) in zip(quads, orientation)]
		for orientation in orientations)
	segments_flattened = ([segment for pair in segments for segment in pair] for segments in segments_oriented)
	costs = (get_max_error_by_angle5_segments(vp, numpy.array(segments))
		for segments in segments_flattened)
	best = min(costs)
	return best

def get_best_intersection_by_angle5_objective(vp, unit_vectors, pivots):
	#print('objective iter:', i[0])
	#print('--- OBJECTIVE')
	#print('vp', repr(vp))
	#print('unit_vectors', repr(unit_vectors))
	#print('pivots', repr(pivots))
	reaches = [pivot - vp for pivot in pivots]
	dists = (numpy.linalg.norm(reach) for reach in reaches)
	unit_reaches = (reach/dist if dist != 0. else numpy.zeros(reach.shape)
		for (reach, dist) in zip(reaches, dists))
	unit_reaches = list(unit_reaches)
	#print('unit_reaches', repr(unit_reaches))
	residuals = [1 - numpy.dot(unit_reach, unit_vector)
		for (unit_vector, unit_reach) in zip(unit_vectors, unit_reaches)]
	#print('residuals', residuals)
	#print('sum-squares', sum(residual**2 for residual in residuals))
	return numpy.array(residuals)

hunts = 0
hunts_inner = 0
def get_best_intersection_by_angle5_segments(vp_seed, segments, tol):
	""" Solve for d = 1 - ((V - P) / |V - P|) . S, and
	del d = (V - P) . S / |V - P|^3 * (V - P) - S / |V - P|"""
	global hunts
	global hunts_inner
	#print('SEGMENTS', segments)
	dims = segments.shape[1]
	vectors = [end - begin for (begin, end) in segments]
	#print('VECTORS', vectors)
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	#print('VECTOR_DISTS', vector_dists)
	# S ~ unit_vectors
	unit_vectors = [vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists)]
	#print('UNIT_VECTORS_INPUT', unit_vectors)
	# FIXME: This recalculates a lot of the same values as other orientations of this same quad
	# P ~ pivots
	pivots = [segment[0] for segment in segments]
	#pivots = [numpy.mean(segment, axis=0) for segment in segments]
	# FIXME: Should this be numpy.eye(dims) ?
	one = numpy.ones(dims)

	seed = numpy.mean(pivots, axis=0) if vp_seed is None else numpy.array(vp_seed)
	context = (unit_vectors, pivots)
	def jacobian(vp, unit_vectors, pivots):
		reaches = [vp - pivot for pivot in pivots]
		mag_reaches = [numpy.linalg.norm(reach) for reach in reaches]
		j = [numpy.dot(reach, unit_vector) / mag_reach**3 * reach - unit_vector / mag_reach
			for (unit_vector, reach, mag_reach) in
				zip(unit_vectors, reaches, mag_reaches)]
		#print('js', numpy.array(j).shape)
		#print('j', j, numpy.array(j).reshape((2, 6)))
		return -numpy.array(j)
	# FIXME
	#jacobian = None
	try:
		#grad_err = scipy.optimize.check_grad(objective, jacobian, seed, *context)
		#print('grad_err', grad_err)
		#result = scipy.optimize.leastsq(get_best_intersection_by_angle5_objective, seed, context, jacobian, col_deriv=True)
		hunts += 1
		hunts_inner += 1
		# FIXME
		#result = scipy.optimize.leastsq(get_best_intersection_by_angle5_objective, seed, context, jacobian, ftol=tol)
		result = scipy.optimize.leastsq(get_best_intersection_by_angle5_objective, seed, context, jacobian, ftol=1e-2, xtol=1e-5)
		#print('leastsq result', result)
		intersect = result[0]
		residuals = get_best_intersection_by_angle5_objective(intersect, unit_vectors, pivots)
		cost = sum(residual**2 for residual in residuals)
		return (cost, intersect)
	except:
		print('exception', sys.exc_info()[0], traceback.format_exc())
		raise

def get_error_by_angle5_segments(vp, segments):
	vectors = [end - begin for (begin, end) in segments]
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	unit_vectors = [vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists)]
	pivots = [segment[0] for segment in segments]
	residuals = get_best_intersection_by_angle5_objective(vp, unit_vectors, pivots)
	cost = sum(residual**2 for residual in residuals)
	#print('get_error_by_angle5_segments', cost, residuals)
	return cost

def get_max_error_by_angle5_segments(vp, segments):
	vectors = [end - begin for (begin, end) in segments]
	vector_dists = (numpy.linalg.norm(vector) for vector in vectors)
	vector_dists = list(vector_dists)
	unit_vectors = [vector / vector_dist if vector_dist != 0. else numpy.zeros(dims)
		for (vector, vector_dist) in zip(vectors, vector_dists)]
	pivots = [segment[0] for segment in segments]
	residuals = get_best_intersection_by_angle5_objective(vp, unit_vectors, pivots)
	cost = max(residuals)
	#print('get_max_error_by_angle5_segments', cost, residuals)
	return cost


def rotate_quad(quad, debug=False):
	"""
	Reorder the points so the upper-leftmost is first
	"""
	#print('quad', quad)
	#center = shapely.geometry.polygon.Polygon(quad).representative_point()
	center = get_centroid(quad)

	angles = (math.atan2(y - center[1], x - center[0]) for (x, y) in quad)
	if debug:
		angles = list(angles)
	# It is assumed the points are already listed counter-clockwise and their sequence should be preserved.
	# Each angle should be 90 degrees greater than the previous angle. All the angles will
	# have the same bias (angular distance from the desired orientation) if the quad is a perfect square.
	bias_angles = [angle + i * (math.pi/2.) for (i, angle) in enumerate(angles)]
	average_bias_angle = math.atan2(
		sum(math.sin(angle) for angle in bias_angles),
		sum(math.cos(angle) for angle in bias_angles))
	# Reorder the points to minimize the bias.
	rotation = int(math.floor(average_bias_angle / (math.pi/2.))) + 2
	#print('rotation', rotation)
	# To rotate to the right, we use the negative of the rotation value
	# To rotate both the (x,y) coordinates together, we multiply by two
	rotated = numpy.roll(quad, -rotation * 2)

	if debug:
		print('--------------------------------')
		print('center', center)
		#print('shifted', (shiftedx, shiftedy))
		#print('new center', (center[0] + transformx, center[1] + transformy))
		print('offsets', [(x - center[0], y - center[1]) for (x, y) in quad])
		print('raw angles', [a / (math.pi/2.) for a in angles])
		print('bias angles', [a / (math.pi/2.) for a in bias_angles])
		print('average bias angle', average_bias_angle / (math.pi/2.))
		print('rotation', rotation)
		print('rotated from', quad)
		print('rotated to', rotated)
		clockwise = numpy.cross(quad[1] - quad[0], quad[2] - quad[1]) >= 0
		print('CLOCKWISE!!!!!!!!!!!!!!!!!!!!!!!' if clockwise else 'counterclockwise')

	return rotated


def get_vps(quads, precalculated=[None,None]):
	vps = (
		get_best_intersection_by_angle3(numpy.array([vector for pair in ([quad[0:2], quad[2:4]] for quad in quads) for vector in pair]), precalculated[0]),
		get_best_intersection_by_angle3(numpy.array([vector for pair in ([quad[1:3], [quad[3], quad[0]]] for quad in quads) for vector in pair]), precalculated[1]),
	)
	return vps


# https://stackoverflow.com/a/45399188
#def reject_outliers_2(data, m = 2.):
def identify_outliers(data, m = 2.):
    d = numpy.abs(data - numpy.median(data))
    mdev = numpy.median(d)
    print('mdev', mdev)
    s = d/(mdev if mdev else 1.)
    #return data[s<m]
    return s<m


if __name__ == "__main__":
	main()
