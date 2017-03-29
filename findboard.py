#!/usr/bin/env python

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


WINNAME = 'Chess Transcription'


def grouper(iterable, n):
	args = [iter(iterable)] * n
	return zip(*args)


def main():

	cv2.namedWindow(WINNAME)

	#webcam = cv2.VideoCapture(0)
	webcam = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')
	#webcam = cv2.VideoCapture('idaho.webm')
	if not webcam.isOpened():
		raise RuntimeError('Failed to open camera')

	[webcam.read() for i in range(10)]
	#[webcam.read() for i in range(1500)]

	if True:
		pattern_size = (7, 7)
		refimgbw = cv2.imread('chessboard.png',0)
		(retval, color2) = webcam.read()
	elif False:
		pattern_size = (6, 9)
		refimgbw = cv2.imread('chessboard96.png',0)
		color2 = cv2.imread('cameracalibrationandposeestimation_sample.jpg')
	else:
		pattern_size = (5, 7)
		refimgbw = cv2.imread('chessboard75.png',0)
		color2 = cv2.imread('chessboard_ready_for_calibration-sm.jpg')

	img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)
	#cv2.imshow(WINNAME, color2)
	#key = cv2.waitKey(0)

	refimg = cv2.cvtColor(refimgbw, cv2.COLOR_GRAY2BGR)
	#cv2.imshow(WINNAME, refimg)
	#key = cv2.waitKey(0)
	(found, ref_chess) = cv2.findChessboardCorners(refimg, pattern_size)
	ref_corners = [c[0] for c in ref_chess]
	#print('chessboard corners', ref_corners)

	#(retval, color1) = webcam.read()
	color1 = numpy.copy(color2)
	img1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)

	#cv2.imshow(WINNAME, img1)
	#key = cv2.waitKey(0)

	#ret,thresh = cv2.threshold(img1,127,255,0)
	contours = []
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

		# For finding dark squares
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
		dilated = cv2.dilate(thresh, kernel)
		# For finding light squares
		eroded = cv2.erode(thresh, kernel)
		#cv2.imshow(WINNAME, dilated)
		#key = cv2.waitKey(0)
		#cv2.imshow(WINNAME, eroded)
		#key = cv2.waitKey(0)

		# FIXME: findChessboardCorners draws a rectangle around the outer edge,
		# so that clipped corners have a chance of being recognized.

		im2, contoursd, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		ime2, contourse, hierarchy = cv2.findContours(eroded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		contoursd_rev = [numpy.array(list(reversed(contour))) for contour in contoursd]

		contours.extend(contoursd_rev)
		contours.extend(contourse)
	#print('contours', len(contours))

	approxes = []
	good = []
	goodd = []
	goode = []
	oldgood = []
	oldgoodd = []
	oldgoode = []
	kernel_cloud = [(kpx - kernel.shape[0]/2, kpy - kernel.shape[1]/2)	
		for kpx in range(kernel.shape[0]) for kpy in range(kernel.shape[1])
		if kernel[(kpx,kpy)]]
	kernel_hull = [p[0] for p in cv2.convexHull(numpy.array(kernel_cloud))]

	#print('kernel_cloud', kernel_cloud)
	for (idx, contour) in enumerate(contours):
		#	contour = numpy.array([[p] for p in contour])
		if len(contour) < 4:
			continue
		perimeter = cv2.arcLength(contour, closed=True)
		# FIXME: The findChessboardCorners utility tries multiple thresholds between 1 and 7,
		# and for each one it does two passes.
		if len(contour) > 4:
			approx = cv2.approxPolyDP(contour, perimeter/20, True)
		elif len(contour) == 4:
			approx = contour
		else:
			continue
		approx = numpy.squeeze(approx)

		#if len(approx) > 4:
		#	approx = cv2.approxPolyDP(approx, 5, True)
		if len(approx) != 4:
			continue
			pass
		# Any negative-oriented (clockwise) contours are rejected.
		if numpy.cross(approx[1] - approx[0], approx[2] - approx[1]) >= 0:
			continue
		if not cv2.isContourConvex(approx):
			continue
		#if cv2.arcLength(approx,True) < 50:
		#	continue
		if cv2.contourArea(approx) < 40:
			continue

		# Remove bookshelf contours
		#if approx[0][0][1] < 300:
		#	continue

		approxes.append(approx)

	for approx in approxes:
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

		good.append(dilated_contour_array)
	#print('GOOD', len(good))

	# Filter out duplicate quads
	tree = scipy.spatial.KDTree([corner for contour in good for corner in contour])
	dist = numpy.linalg.norm([2, 2])
	pairs = tree.query_pairs(dist)
	#print('pairs', pairs)
	connected_indices = collections.defaultdict(lambda: collections.defaultdict(int))
	for pair in pairs:
		connected = set(point_idx // 4 for point_idx in pair)
		for (left, right) in itertools.product(connected, repeat=2):
			if left != right:
				#print(left, '=>', right)
				connected_indices[left][right] += 1
	#print('CONN_IND', len(connected_indices), connected_indices)
	#uniques = {left_idx: {right_idx: right_count for (right_idx, right_count) in right_counts.iteritems()
	#		# If two quads overlap on 3 or more sides, then discard the one that was found
	#		# first, which is the one found in the noiser image with the smaller block size.
	#		if right_idx < left_idx or right_count < 3}
	#	for (left_idx, right_counts) in connected_indices.iteritems()}
	uniques = {left_idx:
		[right_idx
			for (right_idx, right_count) in right_counts.iteritems()
			if right_count < 3]
		for (left_idx, right_counts) in connected_indices.iteritems()
		# If two quads overlap on 3 or more sides, then discard the one that was found
		# first, which is the one found in the noiser image with the smaller block size.
		if all(right_idx < left_idx or right_count < 3 for (right_idx, right_count) in right_counts.iteritems())}
	#print('UNIQUES', len(uniques), uniques)
	connected_map = [left_idx for (left_idx, right_counts) in uniques.iteritems() if right_counts]
	#print('CONN_MAP', len(connected_map), connected_map)
	reverse_connected_map = {old_idx: new_idx for (new_idx, old_idx) in enumerate(connected_map)}
	quads = [good[left_idx] for left_idx in connected_map]
	#print('quads', len(quads))

	contlines = numpy.zeros((color2.shape[0], color2.shape[1], 3), numpy.uint8)
	for contour in good:
		cv2.drawContours(contlines, numpy.array([[(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1)))
		for (x,y) in contour]]), -1, (0, 0, 255), 1)
	for contour in quads:
		cv2.drawContours(contlines, numpy.array([[(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1)))
		for (x,y) in contour]]), -1, (255, 0, 0), 1)
	#cv2.imshow(WINNAME, contlines)
	#key = cv2.waitKey(0)


	color3 = numpy.copy(color2)


	# TODO: Once the board is found, use MSER to track it


	# FIXME: Discard any RANSAC sample sets where the furthest points are more than 9 units apart.
	class ChessboardPerspectiveEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
		def __init__(self, tol=math.pi/36000., shape=(1,1), seed=None):
			self.set_params(tol=tol, shape=shape, seed=seed)
			self.vps = None
		def segment_alignment_error(self, segment, vp, debug=False):
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

			# Try both possible orientations of each quad
			rotated_quad = projected_quad[1:] + projected_quad[:1]
			#if (self.quad_alignment_error(projected_quad, vp1) + self.quad_alignment_error(rotated_quad, vp2) <
			#	self.quad_alignment_error(rotated_quad, vp1) + self.quad_alignment_error(projected_quad, vp2)):
			#	self.quad_alignment_error(projected_quad, vp1, debug=True)
			#	self.quad_alignment_error(rotated_quad, vp2, debug=True)
			#else:
			#	self.quad_alignment_error(rotated_quad, vp1, debug=True)
			#	self.quad_alignment_error(projected_quad, vp2, debug=True)


			distance += min(
				self.quad_alignment_error(projected_quad, vp1) + self.quad_alignment_error(rotated_quad, vp2),
				self.quad_alignment_error(rotated_quad, vp1) + self.quad_alignment_error(projected_quad, vp2))

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

			res = scipy.optimize.minimize(lambda x: optimizer(x), seed, method='L-BFGS-B', tol=tol)
			if not res.success:
				self.vps = None
				return self
			fitted = res.x
			self.vps = numpy.array(grouper(fitted, 2))

			#print('optimization done')

			(vp1, vp2) = (numpy.array(vp) for vp in grouper(fitted, 2))
			working = numpy.copy(color3)
			left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
			right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
			cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in sample_quad]) for sample_quad in sample_quads]), -1, 255, 2)
			cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
			#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
			#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			for quad in sample_quads:
				for point in quad:
					for vp in self.vps:
						cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)
			cv2.imshow(WINNAME, working)
			key = cv2.waitKey(1)

			#print('comparison', self.objective(sample_quads, [360., 200., 6000., 240.], True))

			# FIXME
			# The translation had already been determined deep inside the objective function,
			# but it was lost and needs to be recovered.

			return self
		def score(self, X, y):
			s = super(ChessboardPerspectiveEstimator, self).score(X, y)
			#print('score', s)
			#print('SCORE', X, y, s)
			return s
		def predict(self, X):
			#print('predict')
			quads = [grouper(quad, 2) for quad in X]
			if self.vps is None:
				return [float('inf') for quad in quads]
			predicted = [(self.quad_error(self.vps, quad),) for quad in quads]
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
				(center[0]  % 2 // 1 + center[1] % 2 // 1) % 2),
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
	threshold = sum((dim/16.)**2 for dim in color3.shape[0:2])
	#print('thresh', threshold)
	visible_squares_estimate = 8 * 8 / 4
	success_rate = 0.999999
	retries = int(math.ceil(math.log(1 - success_rate,
		max(0.5, 1 - visible_squares_estimate / float(len(quads))))))
	while retries > 0:
		regressor = sklearn.linear_model.RANSACRegressor(
			base_estimator=ChessboardPerspectiveEstimator(tol=math.pi/360., shape=color3.shape),
			min_samples=3,
			residual_metric=lambda dy: numpy.sum(dy**2, axis=1),
			# Each segment has only 3 degrees of allowed error on average.
			residual_threshold=math.pi/15.,
			#residual_threshold=threshold,
			max_trials=retries,
		)
		#regressor.get_params()['base_estimator'].set_params(tol=math.pi/360., shape=color3.shape)
		# RANSACRegressor expects the input to be an array of points.
		# This target data is an array of quads instead, where each quad
		# contains 4 points. The translation is done by passing all 4 2-D
		# points as if they were a single 8-dimensional point.
		target_pts = [[dim for corner in quad for dim in corner] for quad in quads]
		dark_square = (0., 0., 0., 1., 1., 1., 1., 0.)
		light_square = (1., 0., 1., 1., 2., 1., 2., 0.)
		training_pts = [(0,) for i in range(len(quads))]
		try:
			regressor.fit(target_pts, training_pts)
			break
		except ValueError as e:
			print('Failed regression', e)
		finally:
			retries -= regressor.n_trials_

	(vp1, vp2) = regressor.estimator_.vps
	working = numpy.copy(color3)
	left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
	right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
	inlier_quads = [quad for (quad, mask) in zip(quads, regressor.inlier_mask_) if mask]
	#print('vps', tuple(vp1), tuple(vp2))
	#print('quads', inlier_quads)
	cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in quad]) for quad in inlier_quads]), -1, 255, 2)
	cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
	#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
	#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
	cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
	#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
	#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
	for quad in inlier_quads:
		for point in quad:
			for vp in [vp1, vp2]:
				cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)

	inlier_indices = [idx for (idx, mask) in enumerate(regressor.inlier_mask_) if mask]
	# Discard quads that are no longer touching any other inliers
	filtered_inlier_indices = [idx for idx in inlier_indices
		if any(right_idx in reverse_connected_map and reverse_connected_map[right_idx] in inlier_indices for right_idx in uniques[connected_map[idx]])]
	inlier_quads = [quads[idx] for idx in filtered_inlier_indices]
	#cv2.imshow(WINNAME, working)
	#key = cv2.waitKey(0)
	if len(inlier_quads) < 3:
		raise RuntimeError('Not enough quads found')

	# Run it again on the inliers with higher precision
	estimator = ChessboardPerspectiveEstimator(tol=math.pi/360000., shape=color3.shape, seed=(vp1, vp2))
	target_pts = [[dim for corner in quad for dim in corner] for quad in inlier_quads]
	training_pts = [(0,) for i in range(len(inlier_quads))]
	estimator.fit(target_pts, training_pts)

	(vp1, vp2) = estimator.vps
	working = numpy.copy(color3)
	left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
	right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
	#inlier_quads = [quad for (quad, mask) in zip(inlier_quads, regressor.inlier_mask_) if mask]
	print('vps', tuple(vp1), tuple(vp2))
	#print('quads', len(inlier_quads))
	cv2.drawContours(working, numpy.array([numpy.array([(int(numpy.clip(x, 0, img1.shape[1]-1)), int(numpy.clip(y, 0, img1.shape[0]-1))) for (x,y) in quad]) for quad in inlier_quads]), -1, 255, 2)
	cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 255, 0))
	#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
	#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
	cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 255, 0))
	#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
	#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
	for quad in inlier_quads:
		for point in quad:
			for vp in [vp1, vp2]:
				cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)
	cv2.imshow(WINNAME, working)
	key = cv2.waitKey(0)

	# TODO: Calculate the rotation matrix
	# TODO: Project all quads using the rotation matrix
	# TODO: Find the standard deviation of edge lengths and discard ouliers
	#

	return



	warped = cv2.warpPerspective(refimg, M, (color3.shape[1], color3.shape[0]))
	#warped = skimage.transform.warp(refimg, model, output_shape=color3.shape)
	#warped = numpy.uint8((1-warped_pre)*256)
	#color3[0:warped.shape[0], 0:warped.shape[1]] = warped


	flatmask = numpy.full((refimg.shape[0], refimg.shape[1]), 255, dtype=numpy.uint8)
	flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
	warpedmask = cv2.warpPerspective(flatmask, M, (color3.shape[1], color3.shape[0]))
	#warpedmask = skimage.transform.warp(color3, model)

	#print('WARPED', warped)
	#print('COLOR3', color3)
	maskidx = (warpedmask!=0)
	overlay = numpy.copy(color3)
	overlay[maskidx] = warped[maskidx]
	#cv2.imshow(WINNAME, warped)
	#key = cv2.waitKey(0)

	#warpedpartial = cv2.bitwise_and(warped, warpedmask)
	#invmask = cv2.bitwise_not(warpedmask)
	#color2partial = cv2.bitwise_and(color2, invmask)
	#overlay = cv2.bitwise_or(color2partial, warpedpartial)

	#cv2.imshow(WINNAME, overlay)
	#key = cv2.waitKey(0)

	idealized = cv2.warpPerspective(color3, M, (refimg.shape[1], refimg.shape[0]),
		flags=cv2.WARP_INVERSE_MAP)
	#cv2.imshow(WINNAME, idealized)
	#key = cv2.waitKey(0)


	projected = numpy.copy(overlay)

	axis = numpy.float32([[0,0,0], [(pattern_size[0]+1)*100,0,0], [0,(pattern_size[1]+1)*100,0], [0,0,-100]]).reshape(-1,3)
	axispt, j = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[1].ravel()), (0,0,255), 5)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[2].ravel()), (0,255,0), 5)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[3].ravel()), (255,0,0), 5)

	sqsize = 100
	fillrows = [0, 1, pattern_size[1] - 1, pattern_size[1]]	
	#objp = obj_pts = numpy.array([[(p[0][0], p[0][1], 0) for p in src_pts]]).astype('float32')
	objp = 	obj_pts = numpy.array([[((x+0.5)*sqsize, (y+0.5)*sqsize, 0)
		for x in xrange(pattern_size[0] + 1) for y in fillrows]]).astype('float32')
	#print(objp)
	imgp, j = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
	#objtp = numpy.array([[(p[0][0], p[0][1], -150) for p in src_pts]]).astype('float32')
	objtp = numpy.array([[((x+0.5)*sqsize, (y+0.5)*sqsize, -150)
		for x in xrange(pattern_size[0] + 1) for y in fillrows]]).astype('float32')
	imgtp, j = cv2.projectPoints(objtp, rvec, tvec, mtx, dist)

	for cv in imgp:
		break
		c = cv[0]
		cv2.circle(projected, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))
	for (bp, tp) in zip(imgp, imgtp):
		cv2.line(projected,(bp[0][0],bp[0][1]),(tp[0][0],tp[0][1]),(255,0,0),2)

	#cv2.imshow(WINNAME, projected)
	#key = cv2.waitKey(0)


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


if __name__ == "__main__":
	main()
