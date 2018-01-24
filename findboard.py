#!/usr/bin/env python

from __future__ import print_function

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


WINNAME = 'Chess Transcription'


def grouper(iterable, n):
	args = [iter(iterable)] * n
	return zip(*args)

color_global = None

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
	global color_global
	color_global = numpy.copy(color2)

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
		# FIXME: Start with a 1-pixel dilation and increase to 7, like cv::findChessboardCorners.
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
		#if approx[0][1] < 300:
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
	connected_indices = collections.defaultdict(collections.Counter)
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

			# Try both possible orientations of each quad
			rotated_quad = projected_quad[1:] + projected_quad[:1]
			#if (self.quad_alignment_error(projected_quad, vp1) + self.quad_alignment_error(rotated_quad, vp2) <
			#	self.quad_alignment_error(rotated_quad, vp1) + self.quad_alignment_error(projected_quad, vp2)):
			#	self.quad_alignment_error(projected_quad, vp1, debug=True)
			#	self.quad_alignment_error(rotated_quad, vp2, debug=True)
			#else:
			#	self.quad_alignment_error(rotated_quad, vp1, debug=True)
			#	self.quad_alignment_error(projected_quad, vp2, debug=True)


			#distance += min(
			#	self.quad_alignment_error(projected_quad, vp1) + self.quad_alignment_error(rotated_quad, vp2),
			#	self.quad_alignment_error(rotated_quad, vp1) + self.quad_alignment_error(projected_quad, vp2))
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

	if False:
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
		visible_squares_estimate = 8 * 8 / 4
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
		inlier_quads = [quad for (idx, quad) in enumerate(quads + quads_rot)
			if regressor.inlier_mask_[idx] and idx % len(quads) in filtered_inlier_indices]
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
		key = cv2.waitKey(0)
		inlier_quads = numpy.array(inlier_quads)
	else:

		# FIXME
		(vp1, vp2) = (numpy.array([-2.57592725e+04, 9.18570980e+00]), numpy.array([388.15343615, 177.45470323]))
		inlier_quads = numpy.array(

[[[ 502.        ,  444.        ],
       [ 579.04672897,  445.07009346],
       [ 561.        ,  418.        ],
       [ 491.85106383,  416.93617021]], [[ 351.70042194,  439.99578059],
       [ 426.39754601,  441.04785276],
       [ 422.98818898,  414.90944882],
       [ 354.        ,  417.        ]], [[ 276.9825694 ,  439.04260813],
       [ 287.21153846,  414.03846154],
       [ 220.06482107,  411.90681972],
       [ 201.93357934,  437.9704797 ]], [[ 127.84238806,  438.16477612],
       [ 152.78499665,  412.08841259],
       [  84.12182741,  409.90862944],
       [  52.28699552,  433.78475336]], [[ 561.        ,  418.        ],
       [ 631.93676815,  419.09133489],
       [ 610.        ,  396.        ],
       [ 545.94736842,  396.        ]], [[ 424.        ,  415.        ],
       [ 492.29139073,  416.06705298],
       [ 483.        ,  394.        ],
       [ 420.5       ,  394.        ]], [[ 354.97637795,  414.14173228],
       [ 358.49714286,  393.01714286],
       [ 297.04790419,  390.89820359],
       [ 287.61917808,  410.93424658]], [[ 219.93417722,  413.09620253],
       [ 235.02828467,  391.03558394],
       [ 173.04889741,  389.94822627],
       [ 153.25176678,  410.90989399]], [[ 483. ,  395. ],
       [ 546. ,  395. ],
       [ 534. ,  377. ],
       [ 475.8,  377. ]], [[ 360.        ,  392.        ],
       [ 421.57405405,  393.06162162],
       [ 417.95964126,  373.78475336],
       [ 360.        ,  378.        ]], [[ 296.95166858,  392.10356732],
       [ 305.38778055,  374.02618454],
       [ 248.0438247 ,  372.94422311],
       [ 234.70609756,  389.9195122 ]], [[ 172.89473684,  391.10526316],
       [ 190.94339623,  373.05660377],
       [ 133.07238606,  371.94369973],
       [ 111.31513648,  388.86600496]], [[ 534.        ,  377.        ],
       [ 592.76923077,  377.        ],
       [ 578.        ,  361.        ],
       [ 522.92307692,  361.        ]], [[ 419.        ,  375.        ],
       [ 476.38461538,  375.        ],
       [ 469.        ,  359.        ],
       [ 416.53846154,  359.        ]], [[ 360.98564593,  374.05741627],
       [ 364.74619289,  359.01522843],
       [ 312.03030303,  357.93939394],
       [ 304.51428571,  372.97142857]], [[ 247.95652174,  373.05797101],
       [ 259.21538462,  358.04615385],
       [ 206.06122449,  356.93877551],
       [ 191.05769231,  371.94230769]], [[ 470. ,  360. ],
       [ 523.1,  360. ],
       [ 514. ,  347. ],
       [ 464.8,  347. ]], [[ 310.97126437,  358.06321839],
       [ 317.35      ,  344.03      ],
       [ 268.04183267,  342.93426295],
       [ 259.11641221,  356.95992366]], [[ 205.88593156,  358.12547529],
       [ 218.67206478,  344.06072874],
       [ 169.07964602,  342.93362832],
       [ 153.5785124 ,  355.85123967]], [[ 514.        ,  348.        ],
       [ 564.66666667,  348.        ],
       [ 554.        ,  336.        ],
       [ 506.        ,  336.        ]], [[ 416.        ,  346.        ],
       [ 465.33333333,  346.        ],
       [ 459.96930946,  333.93094629],
       [ 413.25433526,  335.01734104]], [[ 267.94136808,  344.06840391],
       [ 276.46822742,  334.12040134],
       [ 230.13953488,  331.86046512],
       [ 219.06818182,  342.93181818]], [[ 460.        ,  335.        ],
       [ 506.14285714,  335.        ],
       [ 499.        ,  325.        ],
       [ 455.71428571,  325.        ]], [[ 321.93706294,  334.14685315],
       [ 326.25806452,  324.06451613],
       [ 283.08540925,  321.85053381],
       [ 277.33333333,  331.91666667]], [[ 230.93706294,  333.07342657],
       [ 239.45756458,  323.13284133],
       [ 197.17197452,  320.84713376],
       [ 184.71732523,  331.91793313]], [[ 502.08108108,  326.08108108],
       [ 543.51162791,  324.93023256],
       [ 535.        ,  315.        ],
       [ 491.        ,  315.        ]], [[ 413. ,  324. ],
       [ 456.5,  324. ],
       [ 452. ,  315. ],
       [ 410. ,  315. ]], [[ 368.94827586,  324.15517241],
       [ 371.98230088,  315.05309735],
       [ 331.07894737,  312.84210526],
       [ 326.53846154,  321.92307692]], [[ 282.9321267 ,  323.08144796],
       [ 290.44343891,  314.0678733 ],
       [ 249.0678733 ,  312.91855204],
       [ 241.55656109,  321.9321267 ]], [[ 201.96835443,  437.0443038 ],
       [ 219.11678832,  413.03649635],
       [ 156.05797101,  411.94927536],
       [ 128.62893082,  435.94968553]], [[ 492.        ,  416.        ],
       [ 561.16302521,  418.16134454],
       [ 546.        ,  396.        ],
       [ 483.80520733,  394.92767599]], [[ 355.50263158,  412.98421053],
       [ 422.34513274,  415.10619469],
       [ 420.        ,  394.        ],
       [ 359.01714286,  391.89714286]], [[ 287.98130009,  413.04808549],
       [ 296.15876089,  392.02032914],
       [ 235.03468208,  390.94797688],
       [ 221.0212766 ,  411.96808511]], [[ 152.95      ,  411.05      ],
       [ 172.94545455,  391.05454545],
       [ 113.07021277,  389.94574468],
       [  87.20097561,  409.93560976]], [[ 546.        ,  395.        ],
       [ 607.89830508,  396.10532688],
       [ 591.        ,  378.        ],
       [ 533.93506494,  376.9025974 ]], [[ 421.        ,  393.        ],
       [ 482.02352941,  394.07058824],
       [ 476.        ,  376.        ],
       [ 418.59158416,  374.93688119]], [[ 357.98932384,  392.05338078],
       [ 361.6       ,  374.        ],
       [ 305.        ,  374.        ],
       [ 297.72405063,  390.97721519]], [[ 235.        ,  390.        ],
       [ 247.14285714,  373.        ],
       [ 191.        ,  373.        ],
       [ 174.        ,  390.        ]], [[ 480.        ,  376.        ],
       [ 532.75      ,  376.        ],
       [ 524.        ,  361.        ],
       [ 468.85179407,  359.89703588]], [[ 305.  ,  373.  ],
       [ 311.25,  358.  ],
       [ 258.  ,  358.  ],
       [ 248.  ,  373.  ]], [[ 418.51104101,  374.06624606],
       [ 416.        ,  359.        ],
       [ 365.02857143,  356.87619048],
       [ 361.31069364,  372.98699422]], [[ 191.        ,  372.        ],
       [ 204.93478261,  358.06521739],
       [ 154.08108108,  356.93513514],
       [ 135.25      ,  372.        ]], [[ 523.        ,  360.        ],
       [ 576.13043478,  361.13043478],
       [ 564.        ,  349.        ],
       [ 514.92307692,  347.88461538]], [[ 417.        ,  358.        ],
       [ 469.37383178,  359.0911215 ],
       [ 464.        ,  347.        ],
       [ 415.65841584,  345.92574257]], [[ 363.99376299,  358.06237006],
       [ 365.29933481,  345.00665188],
       [ 317.02643172,  343.9339207 ],
       [ 311.80991736,  356.97520661]], [[ 259. ,  357. ],
       [ 266.8,  344. ],
       [ 219. ,  344. ],
       [ 207.3,  357. ]], [[ 465.        ,  346.        ],
       [ 514.34104046,  347.12138728],
       [ 506.        ,  336.        ],
       [ 460.83692308,  334.89846154]], [[ 317.        ,  343.        ],
       [ 321.48192771,  334.03614458],
       [ 276.05136986,  332.92808219],
       [ 268.85714286,  343.        ]], [[ 219.        ,  343.        ],
       [ 230.42857143,  333.        ],
       [ 183.        ,  333.        ],
       [ 171.57142857,  343.        ]], [[ 414.        ,  334.        ],
       [ 459.        ,  334.        ],
       [ 456.        ,  325.        ],
       [ 412.55913978,  323.91397849]], [[ 276.95774648,  333.07394366],
       [ 282.71428571,  323.        ],
       [ 240.        ,  323.        ],
       [ 231.07317073,  331.92682927]], [[ 421.62002153,  393.30678149],
       [ 418.        ,  374.        ],
       [ 362.01408451,  372.94366197],
       [ 358.26556017,  387.93775934]], [[ 578.        ,  444.        ],
       [ 651.71428571,  444.        ],
       [ 630.        ,  420.        ],
       [ 560.94036061,  417.84188627]], [[ 427.        ,  441.        ],
       [ 501.2521228 ,  442.06074461],
       [ 491.        ,  417.        ],
       [ 423.59481583,  414.89358799]], [[ 351.    ,  437.    ],
       [ 354.45  ,  414.    ],
       [ 288.    ,  414.    ],
       [ 277.6875,  438.0625]], [[ 506.        ,  335.        ],
       [ 551.69343066,  336.14233577],
       [ 543.        ,  326.        ],
       [ 500.        ,  326.        ]], [[ 579.04672897,  445.07009346],
       [ 561.        ,  418.        ],
       [ 491.85106383,  416.93617021],
       [ 502.        ,  444.        ]], [[ 426.39754601,  441.04785276],
       [ 422.98818898,  414.90944882],
       [ 354.        ,  417.        ],
       [ 351.70042194,  439.99578059]], [[ 287.21153846,  414.03846154],
       [ 220.06482107,  411.90681972],
       [ 201.93357934,  437.9704797 ],
       [ 276.9825694 ,  439.04260813]], [[ 152.78499665,  412.08841259],
       [  84.12182741,  409.90862944],
       [  52.28699552,  433.78475336],
       [ 127.84238806,  438.16477612]], [[ 631.93676815,  419.09133489],
       [ 610.        ,  396.        ],
       [ 545.94736842,  396.        ],
       [ 561.        ,  418.        ]], [[ 492.29139073,  416.06705298],
       [ 483.        ,  394.        ],
       [ 420.5       ,  394.        ],
       [ 424.        ,  415.        ]], [[ 358.49714286,  393.01714286],
       [ 297.04790419,  390.89820359],
       [ 287.61917808,  410.93424658],
       [ 354.97637795,  414.14173228]], [[ 235.02828467,  391.03558394],
       [ 173.04889741,  389.94822627],
       [ 153.25176678,  410.90989399],
       [ 219.93417722,  413.09620253]], [[ 546. ,  395. ],
       [ 534. ,  377. ],
       [ 475.8,  377. ],
       [ 483. ,  395. ]], [[ 421.57405405,  393.06162162],
       [ 417.95964126,  373.78475336],
       [ 360.        ,  378.        ],
       [ 360.        ,  392.        ]], [[ 305.38778055,  374.02618454],
       [ 248.0438247 ,  372.94422311],
       [ 234.70609756,  389.9195122 ],
       [ 296.95166858,  392.10356732]], [[ 190.94339623,  373.05660377],
       [ 133.07238606,  371.94369973],
       [ 111.31513648,  388.86600496],
       [ 172.89473684,  391.10526316]], [[ 592.76923077,  377.        ],
       [ 578.        ,  361.        ],
       [ 522.92307692,  361.        ],
       [ 534.        ,  377.        ]], [[ 476.38461538,  375.        ],
       [ 469.        ,  359.        ],
       [ 416.53846154,  359.        ],
       [ 419.        ,  375.        ]], [[ 364.74619289,  359.01522843],
       [ 312.03030303,  357.93939394],
       [ 304.51428571,  372.97142857],
       [ 360.98564593,  374.05741627]], [[ 259.21538462,  358.04615385],
       [ 206.06122449,  356.93877551],
       [ 191.05769231,  371.94230769],
       [ 247.95652174,  373.05797101]], [[ 523.1,  360. ],
       [ 514. ,  347. ],
       [ 464.8,  347. ],
       [ 470. ,  360. ]], [[ 317.35      ,  344.03      ],
       [ 268.04183267,  342.93426295],
       [ 259.11641221,  356.95992366],
       [ 310.97126437,  358.06321839]], [[ 218.67206478,  344.06072874],
       [ 169.07964602,  342.93362832],
       [ 153.5785124 ,  355.85123967],
       [ 205.88593156,  358.12547529]], [[ 564.66666667,  348.        ],
       [ 554.        ,  336.        ],
       [ 506.        ,  336.        ],
       [ 514.        ,  348.        ]], [[ 465.33333333,  346.        ],
       [ 459.96930946,  333.93094629],
       [ 413.25433526,  335.01734104],
       [ 416.        ,  346.        ]], [[ 276.46822742,  334.12040134],
       [ 230.13953488,  331.86046512],
       [ 219.06818182,  342.93181818],
       [ 267.94136808,  344.06840391]], [[ 506.14285714,  335.        ],
       [ 499.        ,  325.        ],
       [ 455.71428571,  325.        ],
       [ 460.        ,  335.        ]], [[ 326.25806452,  324.06451613],
       [ 283.08540925,  321.85053381],
       [ 277.33333333,  331.91666667],
       [ 321.93706294,  334.14685315]], [[ 239.45756458,  323.13284133],
       [ 197.17197452,  320.84713376],
       [ 184.71732523,  331.91793313],
       [ 230.93706294,  333.07342657]], [[ 543.51162791,  324.93023256],
       [ 535.        ,  315.        ],
       [ 491.        ,  315.        ],
       [ 502.08108108,  326.08108108]], [[ 456.5,  324. ],
       [ 452. ,  315. ],
       [ 410. ,  315. ],
       [ 413. ,  324. ]], [[ 371.98230088,  315.05309735],
       [ 331.07894737,  312.84210526],
       [ 326.53846154,  321.92307692],
       [ 368.94827586,  324.15517241]], [[ 290.44343891,  314.0678733 ],
       [ 249.0678733 ,  312.91855204],
       [ 241.55656109,  321.9321267 ],
       [ 282.9321267 ,  323.08144796]], [[ 219.11678832,  413.03649635],
       [ 156.05797101,  411.94927536],
       [ 128.62893082,  435.94968553],
       [ 201.96835443,  437.0443038 ]], [[ 561.16302521,  418.16134454],
       [ 546.        ,  396.        ],
       [ 483.80520733,  394.92767599],
       [ 492.        ,  416.        ]], [[ 422.34513274,  415.10619469],
       [ 420.        ,  394.        ],
       [ 359.01714286,  391.89714286],
       [ 355.50263158,  412.98421053]], [[ 296.15876089,  392.02032914],
       [ 235.03468208,  390.94797688],
       [ 221.0212766 ,  411.96808511],
       [ 287.98130009,  413.04808549]], [[ 172.94545455,  391.05454545],
       [ 113.07021277,  389.94574468],
       [  87.20097561,  409.93560976],
       [ 152.95      ,  411.05      ]], [[ 607.89830508,  396.10532688],
       [ 591.        ,  378.        ],
       [ 533.93506494,  376.9025974 ],
       [ 546.        ,  395.        ]], [[ 482.02352941,  394.07058824],
       [ 476.        ,  376.        ],
       [ 418.59158416,  374.93688119],
       [ 421.        ,  393.        ]], [[ 361.6       ,  374.        ],
       [ 305.        ,  374.        ],
       [ 297.72405063,  390.97721519],
       [ 357.98932384,  392.05338078]], [[ 247.14285714,  373.        ],
       [ 191.        ,  373.        ],
       [ 174.        ,  390.        ],
       [ 235.        ,  390.        ]], [[ 532.75      ,  376.        ],
       [ 524.        ,  361.        ],
       [ 468.85179407,  359.89703588],
       [ 480.        ,  376.        ]], [[ 311.25,  358.  ],
       [ 258.  ,  358.  ],
       [ 248.  ,  373.  ],
       [ 305.  ,  373.  ]], [[ 416.        ,  359.        ],
       [ 365.02857143,  356.87619048],
       [ 361.31069364,  372.98699422],
       [ 418.51104101,  374.06624606]], [[ 204.93478261,  358.06521739],
       [ 154.08108108,  356.93513514],
       [ 135.25      ,  372.        ],
       [ 191.        ,  372.        ]], [[ 576.13043478,  361.13043478],
       [ 564.        ,  349.        ],
       [ 514.92307692,  347.88461538],
       [ 523.        ,  360.        ]], [[ 469.37383178,  359.0911215 ],
       [ 464.        ,  347.        ],
       [ 415.65841584,  345.92574257],
       [ 417.        ,  358.        ]], [[ 365.29933481,  345.00665188],
       [ 317.02643172,  343.9339207 ],
       [ 311.80991736,  356.97520661],
       [ 363.99376299,  358.06237006]], [[ 266.8,  344. ],
       [ 219. ,  344. ],
       [ 207.3,  357. ],
       [ 259. ,  357. ]], [[ 514.34104046,  347.12138728],
       [ 506.        ,  336.        ],
       [ 460.83692308,  334.89846154],
       [ 465.        ,  346.        ]], [[ 321.48192771,  334.03614458],
       [ 276.05136986,  332.92808219],
       [ 268.85714286,  343.        ],
       [ 317.        ,  343.        ]], [[ 230.42857143,  333.        ],
       [ 183.        ,  333.        ],
       [ 171.57142857,  343.        ],
       [ 219.        ,  343.        ]], [[ 459.        ,  334.        ],
       [ 456.        ,  325.        ],
       [ 412.55913978,  323.91397849],
       [ 414.        ,  334.        ]], [[ 282.71428571,  323.        ],
       [ 240.        ,  323.        ],
       [ 231.07317073,  331.92682927],
       [ 276.95774648,  333.07394366]], [[ 418.        ,  374.        ],
       [ 362.01408451,  372.94366197],
       [ 358.26556017,  387.93775934],
       [ 421.62002153,  393.30678149]], [[ 651.71428571,  444.        ],
       [ 630.        ,  420.        ],
       [ 560.94036061,  417.84188627],
       [ 578.        ,  444.        ]], [[ 501.2521228 ,  442.06074461],
       [ 491.        ,  417.        ],
       [ 423.59481583,  414.89358799],
       [ 427.        ,  441.        ]], [[ 354.45  ,  414.    ],
       [ 288.    ,  414.    ],
       [ 277.6875,  438.0625],
       [ 351.    ,  437.    ]], [[ 551.69343066,  336.14233577],
       [ 543.        ,  326.        ],
       [ 500.        ,  326.        ],
       [ 506.        ,  335.        ]]]
		)


	# Calculate the camera parameters
	# See https://fedcsis.org/proceedings/2012/pliks/110.pdf
	horizon = vp2 - vp1
	horizon_norm = numpy.linalg.norm(horizon)
	unit_horizon = horizon / horizon_norm if horizon_norm != 0. else numpy.zeros(horizon.shape)
	image_dim = numpy.array([color3.shape[1], color3.shape[0]])
	oi = image_dim / 2.
	oi_projection = numpy.dot(oi - vp1, unit_horizon)
	vi = vp1 + oi_projection * unit_horizon
	#print('vi', vi)
	square_f = numpy.linalg.norm(vi - vp1) * numpy.linalg.norm(vi - vp2) - numpy.linalg.norm(vi - oi)**2
	if square_f > 0:
		f = math.sqrt(square_f)
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
	default_mtx = numpy.array([[fx, 0, w/2.], [0, fy, h/2.], [0, 0, 1]]).astype('float32')
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
	cv2.imshow(WINNAME, bg)
	key = cv2.waitKey(1)

	tvec = numpy.array([0., 0., 1.])
	def reverse_project(p):
		proj = numpy.dot(numpy.linalg.inv(numpy.dot(default_mtx, numpy.concatenate([numpy.delete(r, 2, 1), tvec.reshape(3,1)], axis=1))), numpy.array(p).reshape(3,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]

	bg = numpy.copy(color3)
	rq = []
	for quad in inlier_quads:
		quadpts = numpy.array([reverse_project([pt[0], pt[1], 1.]) for pt in quad])
		rq.append(quadpts)
		plot = numpy.array([[x * 30000 + 2000, y * 30000 + 300] for (x,y) in quadpts]).astype('int')
		#print('reverse projected quad', plot, get_perimeter(quadpts))
		cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		cv2.drawContours(bg, [plot], -1, (255, 0, 0), 2)
	projected_quads = list(rq)
	cv2.imshow(WINNAME, bg)
	key = cv2.waitKey(1)

	# FIXME: Calculate the x and y deltas separately (ignoring lines' slant)
	# once it is known that all quads are oriented consistently!

	#print('perimeters-first', [get_perimeter(quad) for quad in rq])
	perimeter = sum(get_perimeter(quad) for quad in rq)
	avg = perimeter / float(len(rq))
	#print('avg-first', avg, len(rq))
	THRESHOLD = avg * 0.1
	#print('perimeter errors', [abs(avg - get_perimeter(quad)) for quad in rq])
	rq_inliers = [quad for quad in rq if abs(avg - get_perimeter(quad)) < THRESHOLD]
	#print('perimeters-second', [get_perimeter(quad) for quad in rq_inliers])
	perimeter = sum(get_perimeter(quad) for quad in rq_inliers)
	avg = perimeter / float(len(rq_inliers))
	#print('avg-second', avg, len(rq_inliers))
	square = avg / 4.

	for quadpts in rq_inliers:
		plot = numpy.array([[x * 30000 + 2000, y * 30000 + 300] for (x,y) in quadpts]).astype('int')
		cv2.drawContours(bg, [plot], -1, (0, 0, 255), 1)
	cv2.imshow(WINNAME, bg)
	key = cv2.waitKey(1)


	# We need the best translation that puts the points closest to true square points.
	# The translation in each dimension can be calculated independently.
	# It doesn't matter which square the points are near, so that means that modular arithmetic is needed.
	# The offsets need to be brought as close as possible to any integral values.
	# To do that, the points are translated to points on a unit circle and the average angle is computed.
	# TODO: This could distinguish between light and dark squares
	biased_points = [[dim / square for dim in p] for quad in rq_inliers for p in quad]
	angles = ([dim * math.pi*2 for dim in p] for p in biased_points)
	(anglesx, anglesy) = zip(*angles)
	biasanglex = math.atan2(
		sum(math.sin(angle) for angle in anglesx),
		sum(math.cos(angle) for angle in anglesx))
	biasangley = math.atan2(
		sum(math.sin(angle) for angle in anglesy),
		sum(math.cos(angle) for angle in anglesy))
	biasx = biasanglex / (math.pi*2)
	biasy = biasangley / (math.pi*2)
	translation = numpy.array([-biasx, -biasy, 0.])
	shifted_points = ([x - biasx, y - biasy] for (x, y) in biased_points)
	#print('translation', translation)
	# FIXME: Then remove outliers and recalculate.
	# FIXME: The numbers before the round should be nearly whole numbers!
	shifted_points = list(shifted_points)
	#print('before round', shifted_points)
	grid_positions = [[int(round(dim)) for dim in p] for p in shifted_points]
	min_position = (min(p[0] for p in grid_positions), min(p[1] for p in grid_positions))
	max_position = (max(p[0] for p in grid_positions), max(p[1] for p in grid_positions))
	padded_begin_position = (min_position[0] - 7, min_position[1] - 7)
	padded_end_position = (max_position[0] + 8, max_position[1] + 8)
	grid_corners = itertools.product(
		range(padded_begin_position[0], padded_end_position[0] - padded_begin_position[0]),
		range(padded_begin_position[1], padded_end_position[1] - padded_begin_position[1]))
	grid_corners = list(grid_corners)
	grid_corners_coord = (numpy.array([x, y, 1.]).reshape(3,1) for (x, y) in grid_corners)
	zoom_in = numpy.array([[square, 0., 0.], [0., square, 0.], [0., 0., 1.]])
	scaled = numpy.dot(r, zoom_in)
	#print('square', square)
	offsetM = numpy.array([[1., 0., biasx], [0., 1., biasy], [0., 0., 1.]])
	reverse_homography_denorm = default_mtx.dot(
			numpy.delete(numpy.concatenate([scaled, tvec.reshape(3,1)], axis=1), 2, 1)
		).dot(offsetM)
	reverse_homography = reverse_homography_denorm / reverse_homography_denorm[2][2]
	#print('reverse', reverse_homography)
	projected_grid_corners_denorm = (reverse_homography.dot(coord) for coord in grid_corners_coord)
	project_grid_points = (coord.reshape(3)[:2] / coord[2] for coord in projected_grid_corners_denorm)

	all_quad_pts = []
	for quad in inlier_quads:
		quadpts = numpy.array([reverse_project([pt[0], pt[1], 1.]) for pt in quad])
		all_quad_pts.extend(quadpts)
	quad_pts_center = ((max(x for (x,y) in all_quad_pts)+min(x for (x,y) in all_quad_pts))/2.,
		(max(y for (x,y) in all_quad_pts)+min(y for (x,y) in all_quad_pts))/2.)
	quad_pts_coef = min(bg.shape[1] / (max(x for (x,y) in all_quad_pts) - min(x for (x,y) in all_quad_pts)),
		bg.shape[0] / (max(y for (x,y) in all_quad_pts) - min(y for (x,y) in all_quad_pts)))


	#print(projected_quads)

	# Find the average square size
	#perimeters = numpy.array([get_perimeter(quad) for quad in projected_quads])
	#outlier_perimeter_filter = identify_outliers(perimeters)
	#filtered_perimeters = perimeters[outlier_perimeter_filter]
	#avg_side = sum(filtered_perimeters) / float(len(filtered_perimeters) * 4)
	#filtered_quads = numpy.array(projected_quads)[outlier_perimeter_filter]
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
	unprojected_quads = numpy.array(inlier_quads)[outlier_perimeter_filter]
	#print('AVG_SIDE', avg_x_side, avg_y_side)

	x_values = (x for quad in filtered_quads for (x, y) in quad)
	y_values = (y for quad in filtered_quads for (x, y) in quad)
	# Plot the points on a unit circle, where it wraps around again every avg_side.
	x_radians = [x * 2*math.pi / avg_x_side for x in x_values]
	y_radians = [y * 2*math.pi / avg_y_side for y in y_values]
	# Now average all the vectors on the unit circle to find an average that isn't fooled by the wrapping.
	offset_angle_x = math.atan2(
		sum(math.sin(angle) for angle in x_radians),
		sum(math.cos(angle) for angle in x_radians))
	offset_angle_y = math.atan2(
		sum(math.sin(angle) for angle in y_radians),
		sum(math.cos(angle) for angle in y_radians))
	# Use that to get the average offset
	avg_x_offset = offset_angle_x / (2*math.pi)
	avg_y_offset = offset_angle_y / (2*math.pi)
	# Shift all the quads so they should now lie on an integral grid
	transformed_quads = [[(x / avg_x_side - avg_x_offset, y / avg_y_side - avg_y_offset) for (x, y) in quad] for quad in filtered_quads]
	#print("TRANSFORMED", transformed_quads)
	# Snap all points to the nearest grid position
	snapped_quads = [[(round(x), round(y)) for (x, y) in quad] for quad in transformed_quads]
	all_snapped_pts = [p for quad in snapped_quads for p in quad]
	all_transformed_pts = (p for quad in transformed_quads for p in quad)
	snap_distance = (numpy.linalg.norm([sp[0] - tp[0], sp[1] - tp[1]]) for (sp, tp) in zip(all_snapped_pts, all_transformed_pts))
	outlier_snap_filter = [d < 1/8. for d in snap_distance]
	snapped_pts = list(itertools.compress(all_snapped_pts, outlier_snap_filter))
	transformed_pts = list(itertools.compress((p for quad in transformed_quads for p in quad), outlier_snap_filter))

	# Match the snapped points with their original image locations
	unprojected_pts_with_outliers = unprojected_quads.reshape((len(unprojected_quads)*4, 2))
	unprojected_pts = unprojected_pts_with_outliers[outlier_snap_filter]
	snapped_pts_coord = numpy.array([[x, y, 1.] for (x, y) in snapped_pts])

	(h, w, _) = color3.shape
	f = max(h, w)
	fx = fy = f
	default_mtx = numpy.array([[fx, 0, w/2.], [0, fy, h/2.], [0, 0, 1]]).astype('float32')
	# TODO: Supply the previously-found rvec and tvec as an optimization
	s, rvecs, tvecs = cv2.solvePnP(snapped_pts_coord, unprojected_pts, default_mtx, dist)
	inliers = [[i] for i in xrange(len(unprojected_pts))]

	min_snapped_x = min(x for (x, y) in snapped_pts) - 7
	min_snapped_y = min(y for (x, y) in snapped_pts) - 7
	max_snapped_x = max(x for (x, y) in snapped_pts) + 7
	max_snapped_y = max(y for (x, y) in snapped_pts) + 7
	grid_corners = list(itertools.product(
		range(int(min_snapped_x), int(max_snapped_x) + 1),
		range(int(min_snapped_y), int(max_snapped_y) + 1)))
	grid_corners_coord = numpy.array([[float(x), float(y), 1.] for (x, y) in grid_corners])
	project_grid_points, j = cv2.projectPoints(grid_corners_coord, rvecs, tvecs, default_mtx, dist)
	#print('GRID', min_snapped_x, max_snapped_x, min_snapped_y, max_snapped_y)


	bg = numpy.copy(color3)
	for quadpts in projected_quads:
		plot = numpy.array([[(x - quad_pts_center[0]) * quad_pts_coef + bg.shape[1]/2., (y - quad_pts_center[1]) * quad_pts_coef + bg.shape[0]/2.] for (x,y) in quadpts]).astype('int')
		#print('reverse projected quad', plot, get_perimeter(quadpts))
		#cv2.drawContours(bg, [plot], -1, (255, 0, 0), 2)
	all_quad_pts = [p for quad in transformed_quads for p in quad]
	quad_pts_center = ((max(x for (x,y) in all_quad_pts)+min(x for (x,y) in all_quad_pts))/2.,
		(max(y for (x,y) in all_quad_pts)+min(y for (x,y) in all_quad_pts))/2.)
	quad_pts_coef = min(bg.shape[1] / (max(x for (x,y) in all_quad_pts) - min(x for (x,y) in all_quad_pts)),
		bg.shape[0] / (max(y for (x,y) in all_quad_pts) - min(y for (x,y) in all_quad_pts)))
	for quadpts in transformed_quads:
		plot = numpy.array([[(x - quad_pts_center[0]) * quad_pts_coef + bg.shape[1]/2., (y - quad_pts_center[1]) * quad_pts_coef + bg.shape[0]/2.] for (x,y) in quadpts]).astype('int')
		#print('reverse projected quad', plot, get_perimeter(quadpts))
		#cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		cv2.drawContours(bg, [plot], -1, (255, 0, 255), 2)
		#for corner in plot:
		#	cv2.circle(bg, tuple(corner), random.randint(1, 24), (0, 255, 255))
		#cv2.imshow(WINNAME, bg)
		#key = cv2.waitKey(100)
	for (x, y) in transformed_pts:
		pt = (int((x - quad_pts_center[0]) * quad_pts_coef + bg.shape[1]/2.), int((y - quad_pts_center[1]) * quad_pts_coef + bg.shape[0]/2.))
		#print('reverse projected quad', plot, get_perimeter(quadpts))
		#cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		#cv2.circle(bg, tuple(pt), random.randint(1, 7), (0, 255, 255))
		cv2.circle(bg, tuple(pt), 2, (0, 255, 255))

	for c in grid_corners:
		plot = (int((c[0] - quad_pts_center[0]) * quad_pts_coef + bg.shape[1]/2.), int((c[1] - quad_pts_center[1]) * quad_pts_coef + bg.shape[0]/2.))
		if plot[0] >= 0 and plot[1] >= 0:
			cv2.circle(bg, plot, 2, (0, 255, 0))
	for quadpts in snapped_quads:
		plot = numpy.array([[(x - quad_pts_center[0]) * quad_pts_coef + bg.shape[1]/2., (y - quad_pts_center[1]) * quad_pts_coef + bg.shape[0]/2.] for (x,y) in quadpts]).astype('int')
		cv2.drawContours(bg, [plot], -1, (255, 0, 0), 1)
	#print(quad_pts_center, bg.shape)
	cv2.imshow(WINNAME, bg)
	key = cv2.waitKey(1)



	pimg = numpy.copy(color3)
	for quad in inlier_quads:
		cv2.drawContours(pimg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
	snapped_pts_coord = numpy.array([[x, y, 1.] for (x, y) in snapped_pts])
	project_snapped_points, j = cv2.projectPoints(snapped_pts_coord, rvecs, tvecs, default_mtx, dist)
	for cp in project_grid_points:
		cv2.circle(pimg, (int(round(cp[0][0])), int(round(cp[0][1]))), 1, (0, 255, 0))
	inlier_set = frozenset(idx[0] for idx in inliers)
	for (idx, (cu, cp)) in enumerate(zip(unprojected_pts, project_snapped_points)):
		#if idx not in inlier_set:
		#	cv2.circle(pimg, (int(round(cu[0])), int(round(cu[1]))), 4, (255, 0, 0))
		#	continue
		cv2.line(pimg, (int(round(cu[0])), int(round(cu[1]))), (int(round(cp[0][0])), int(round(cp[0][1]))), (255,0,0), 1)
		cv2.circle(pimg, (int(round(cu[0])), int(round(cu[1]))), 4, (0, 255, 255))
		cv2.circle(pimg, (int(round(cp[0][0])), int(round(cp[0][1]))), 2, (0, 255, 0))
	cv2.imshow(WINNAME, pimg)
	key = cv2.waitKey(1)


	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	result = cv2.cornerSubPix(img1, numpy.float32(project_grid_points), (4,4), (-1,-1), criteria)
	pimg = numpy.copy(color3)
	for cp in project_grid_points:
		cv2.circle(pimg, (int(round(cp[0][0])), int(round(cp[0][1]))), 1, (0, 255, 0))
	for cp in result:
		cv2.circle(pimg, (int(round(cp[0][0])), int(round(cp[0][1]))), 2, (0, 255, 255))
	cv2.imshow(WINNAME, pimg)
	key = cv2.waitKey(0)


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
	pairs = itertools.izip(left, itertools.chain(right, wrap))
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
		print('hypotheses', len(hypotheses), '{}/{}'.format(idx, len(quads)))
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
	Transform the points by an integral distance and
	reorder the points so the upper-leftmost is first
	"""
	#print('quad', quad)
	#center = shapely.geometry.polygon.Polygon(quad).representative_point()
	center = get_centroid(quad)

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
	if rotation > 0:
		rotated = quad[rotation:] + quad[:rotation]
	else:
		rotated = quad

	if debug:
		print('--------------------------------')
		print('center', center)
		#print('shifted', (shiftedx, shiftedy))
		#print('transform', (transformx, transformy))
		#print('new center', (center[0] + transformx, center[1] + transformy))
		print('offsets', [(x - center[0], y - center[1]) for (x, y) in quad])
		print('raw angles', [a / (math.pi/2.) for a in angles])
		print('bias angles', [a / (math.pi/2.) for a in bias_angles])
		print('average bias angle', average_bias_angle / (math.pi/2.))
		print('rotation', rotation)
		#print('rotated from', transformed)
		print('rotated from', quad)
		print('rotated to', rotated)
		if center[0] + transformx >= 2:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
    s = d/(mdev if mdev else 1.)
    #return data[s<m]
    return s<m


if __name__ == "__main__":
	main()
