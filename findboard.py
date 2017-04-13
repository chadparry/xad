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


WINNAME = 'Chess Transcription'


def grouper(iterable, n):
	args = [iter(iterable)] * n
	return zip(*args)


def main():

	cv2.namedWindow(WINNAME)

	#webcam = cv2.VideoCapture(0)
	#webcam = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')
	webcam = cv2.VideoCapture('idaho.webm')
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
			cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 1)
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

	if True:
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
		cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 1)
		for quad in inlier_quads:
			for point in quad:
				for vp in [vp1, vp2]:
					cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)

		inlier_indices = [idx for (idx, mask) in enumerate(regressor.inlier_mask_) if mask]
		# Discard quads that are no longer touching any other inliers
		filtered_inlier_indices = [idx for idx in inlier_indices
			if any(right_idx in reverse_connected_map and reverse_connected_map[right_idx] in inlier_indices for right_idx in uniques[connected_map[idx]])]
		inlier_quads = [quads[idx] for idx in filtered_inlier_indices]
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
		cv2.line(working, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 1)
		for quad in inlier_quads:
			for point in quad:
				for vp in [vp1, vp2]:
					cv2.line(working, tuple(int(d) for d in point), tuple(int(d) for d in vp), (0,0,255), 1)
		#cv2.imshow(WINNAME, working)
		#key = cv2.waitKey(0)
	else:

		# FIXME
		(vp1, vp2) = (numpy.array((392.6853257874306, 169.31476831226928)), numpy.array((53355.770532385861, 1499.2484281880058)))
		inlier_quads = numpy.array(
[[[ 421.62002153,  393.30678149,    ],
  [ 418.,          374.,            ],
  [ 362.01408451,  372.94366197,    ],
  [ 358.26556017,  387.93775934,    ],],

 [[ 427.,          441.,            ],
  [ 501.2521228,   442.06074461,    ],
  [ 491.,          417.,            ],
  [ 423.59481583,  414.89358799,    ],],

 [[ 351.,          437.,            ],
  [ 354.45,        414.,            ],
  [ 288.,          414.,            ],
  [ 277.6875,      438.0625,        ],],

 [[ 506.,          335.,            ],
  [ 551.69343066,  336.14233577,    ],
  [ 543.,          326.,            ],
  [ 500.,          326.,            ],],

 [[ 502.,          444.,            ],
  [ 579.04672897,  445.07009346,    ],
  [ 561.,          418.,            ],
  [ 491.85106383,  416.93617021,    ],],

 [[ 351.70042194,  439.99578059,    ],
  [ 426.39754601,  441.04785276,    ],
  [ 422.98818898,  414.90944882,    ],
  [ 354.,          417.,            ],],

 [[ 276.9825694,   439.04260813,    ],
  [ 287.21153846,  414.03846154,    ],
  [ 220.06482107,  411.90681972,    ],
  [ 201.93357934,  437.9704797,     ],],

 [[ 127.84238806,  438.16477612,    ],
  [ 152.78499665,  412.08841259,    ],
  [  84.12182741,  409.90862944,    ],
  [  52.28699552,  433.78475336,    ],],

 [[ 561.,          418.,            ],
  [ 631.93676815,  419.09133489,    ],
  [ 610.,          396.,            ],
  [ 545.94736842,  396.,            ],],

 [[ 424.,          415.,            ],
  [ 492.29139073,  416.06705298,    ],
  [ 483.,          394.,            ],
  [ 420.5,         394.,            ],],

 [[ 354.97637795,  414.14173228,    ],
  [ 358.49714286,  393.01714286,    ],
  [ 297.04790419,  390.89820359,    ],
  [ 287.61917808,  410.93424658,    ],],

 [[ 219.93417722,  413.09620253,    ],
  [ 235.02828467,  391.03558394,    ],
  [ 173.04889741,  389.94822627,    ],
  [ 153.25176678,  410.90989399,    ],],

 [[ 483.,          395.,            ],
  [ 546.,          395.,            ],
  [ 534.,          377.,            ],
  [ 475.8,         377.,            ],],

 [[ 360.,          392.,            ],
  [ 421.57405405,  393.06162162,    ],
  [ 417.95964126,  373.78475336,    ],
  [ 360.,          378.,            ],],

 [[ 296.95166858,  392.10356732,    ],
  [ 305.38778055,  374.02618454,    ],
  [ 248.0438247,   372.94422311,    ],
  [ 234.70609756,  389.9195122,     ],],

 [[ 172.89473684,  391.10526316,    ],
  [ 190.94339623,  373.05660377,    ],
  [ 133.07238606,  371.94369973,    ],
  [ 111.31513648,  388.86600496,    ],],

 [[ 534.,          377.,            ],
  [ 592.76923077,  377.,            ],
  [ 578.,          361.,            ],
  [ 522.92307692,  361.,            ],],

 [[ 419.,          375.,            ],
  [ 476.38461538,  375.,            ],
  [ 469.,          359.,            ],
  [ 416.53846154,  359.,            ],],

 [[ 360.98564593,  374.05741627,    ],
  [ 364.74619289,  359.01522843,    ],
  [ 312.03030303,  357.93939394,    ],
  [ 304.51428571,  372.97142857,    ],],

 [[ 247.95652174,  373.05797101,    ],
  [ 259.21538462,  358.04615385,    ],
  [ 206.06122449,  356.93877551,    ],
  [ 191.05769231,  371.94230769,    ],],

 [[ 470.,          360.,            ],
  [ 523.1,         360.,            ],
  [ 514.,          347.,            ],
  [ 464.8,         347.,            ],],

 [[ 310.97126437,  358.06321839,    ],
  [ 317.35,        344.03,          ],
  [ 268.04183267,  342.93426295,    ],
  [ 259.11641221,  356.95992366,    ],],

 [[ 205.88593156,  358.12547529,    ],
  [ 218.67206478,  344.06072874,    ],
  [ 169.07964602,  342.93362832,    ],
  [ 153.5785124,   355.85123967,    ],],

 [[ 514.,          348.,            ],
  [ 564.66666667,  348.,            ],
  [ 554.,          336.,            ],
  [ 506.,          336.,            ],],

 [[ 267.94136808,  344.06840391,    ],
  [ 276.46822742,  334.12040134,    ],
  [ 230.13953488,  331.86046512,    ],
  [ 219.06818182,  342.93181818,    ],],

 [[ 321.93706294,  334.14685315,    ],
  [ 326.25806452,  324.06451613,    ],
  [ 283.08540925,  321.85053381,    ],
  [ 277.33333333,  331.91666667,    ],],

 [[ 230.93706294,  333.07342657,    ],
  [ 239.45756458,  323.13284133,    ],
  [ 197.17197452,  320.84713376,    ],
  [ 184.71732523,  331.91793313,    ],],

 [[ 282.9321267,   323.08144796,    ],
  [ 290.44343891,  314.0678733,     ],
  [ 249.0678733,   312.91855204,    ],
  [ 241.55656109,  321.9321267,     ],],

 [[ 201.96835443,  437.0443038,     ],
  [ 219.11678832,  413.03649635,    ],
  [ 156.05797101,  411.94927536,    ],
  [ 128.62893082,  435.94968553,    ],],

 [[ 492.,          416.,            ],
  [ 561.16302521,  418.16134454,    ],
  [ 546.,          396.,            ],
  [ 483.80520733,  394.92767599,    ],],

 [[ 355.50263158,  412.98421053,    ],
  [ 422.34513274,  415.10619469,    ],
  [ 420.,          394.,            ],
  [ 359.01714286,  391.89714286,    ],],

 [[ 287.98130009,  413.04808549,    ],
  [ 296.15876089,  392.02032914,    ],
  [ 235.03468208,  390.94797688,    ],
  [ 221.0212766,   411.96808511,    ],],

 [[ 152.95,        411.05,          ],
  [ 172.94545455,  391.05454545,    ],
  [ 113.07021277,  389.94574468,    ],
  [  87.20097561,  409.93560976,    ],],

 [[ 546.,          395.,            ],
  [ 607.89830508,  396.10532688,    ],
  [ 591.,          378.,            ],
  [ 533.93506494,  376.9025974,     ],],

 [[ 421.,          393.,            ],
  [ 482.02352941,  394.07058824,    ],
  [ 476.,          376.,            ],
  [ 418.59158416,  374.93688119,    ],],

 [[ 357.98932384,  392.05338078,    ],
  [ 361.6,         374.,            ],
  [ 305.,          374.,            ],
  [ 297.72405063,  390.97721519,    ],],

 [[ 235.,          390.,            ],
  [ 247.14285714,  373.,            ],
  [ 191.,          373.,            ],
  [ 174.,          390.,            ],],

 [[ 480.,          376.,            ],
  [ 532.75,        376.,            ],
  [ 524.,          361.,            ],
  [ 468.85179407,  359.89703588,    ],],

 [[ 305.,          373.,            ],
  [ 311.25,        358.,            ],
  [ 258.,          358.,            ],
  [ 248.,          373.,            ],],

 [[ 418.51104101,  374.06624606,    ],
  [ 416.,          359.,            ],
  [ 365.02857143,  356.87619048,    ],
  [ 361.31069364,  372.98699422,    ],],

 [[ 191.,          372.,            ],
  [ 204.93478261,  358.06521739,    ],
  [ 154.08108108,  356.93513514,    ],
  [ 135.25,        372.,            ],],

 [[ 523.,          360.,            ],
  [ 576.13043478,  361.13043478,    ],
  [ 564.,          349.,            ],
  [ 514.92307692,  347.88461538,    ],],

 [[ 417.,          358.,            ],
  [ 469.37383178,  359.0911215,     ],
  [ 464.,          347.,            ],
  [ 415.65841584,  345.92574257,    ],],

 [[ 363.99376299,  358.06237006,    ],
  [ 365.29933481,  345.00665188,    ],
  [ 317.02643172,  343.9339207,     ],
  [ 311.80991736,  356.97520661,    ],],

 [[ 259.,          357.,            ],
  [ 266.8,         344.,            ],
  [ 219.,          344.,            ],
  [ 207.3,         357.,            ],],

 [[ 465.,          346.,            ],
  [ 514.34104046,  347.12138728,    ],
  [ 506.,          336.,            ],
  [ 460.83692308,  334.89846154,    ],],

 [[ 317.,          343.,            ],
  [ 321.48192771,  334.03614458,    ],
  [ 276.05136986,  332.92808219,    ],
  [ 268.85714286,  343.,            ],],

 [[ 219.,          343.,            ],
  [ 230.42857143,  333.,            ],
  [ 183.,          333.,            ],
  [ 171.57142857,  343.,            ],],

 [[ 414.,          334.,            ],
  [ 459.,          334.,            ],
  [ 456.,          325.,            ],
  [ 412.55913978,  323.91397849,    ],],

 [[ 276.95774648,  333.07394366,    ],
  [ 282.71428571,  323.,            ],
  [ 240.,          323.,            ],
  [ 231.07317073,  331.92682927,    ],],],

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
	print('vi', vi)
	square_f = numpy.linalg.norm(vi - vp1) * numpy.linalg.norm(vi - vp2) - numpy.linalg.norm(vi - oi)**2
	if square_f > 0:
		f = math.sqrt(square_f)
	else:
		print("Camera center is not aligned with image center", file=sys.stderr)
		f = max(image_dim)
	print('f', f)

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
	print('x', unit_x_axis)
	print('y', unit_y_axis)
	print('z', unit_z_axis)
	rdenorm = numpy.array([
		[vp1[0] / denom1, vp2[0] / denom2, unit_z_axis[0]],
		[vp1[1] / denom1, vp2[1] / denom2, unit_z_axis[1]],
		[f / denom1, f / denom2, unit_z_axis[2]],
	])
	r = rdenorm / rdenorm[2][2]
	print('R', r)



	(h, w, _) = color3.shape
	#f = max(h, w)
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

	# Huge squares:
	#tvec = numpy.array([-1., 0., 0.])
	#zoom_out = numpy.array([[0.05, 0., 0.], [0., 0.1, 0.], [0., 0., 1.]])
	#tvec = numpy.array([-10., 10., 0.])
	#zoom_out = numpy.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
	tvec = numpy.array([-100., 50., 0.])
	zoom_out = numpy.array([[3., 0., 0.], [0., 10., 0.], [0., 0., 1.]])

	r = numpy.dot(r, zoom_out)
	r = r / r[2][2]

	print('R', r)
	#print('tvec', tvec.reshape(3,1))

	rt1 = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array([1., 0., 0., 0.]).reshape(4,1))
	rt2 = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array([0., 1., 0., 0.]).reshape(4,1))
	print('round trip 1', vp1, '->', (rt1.reshape(1,3) / rt1[2][0])[0][0:2])
	print('round trip 2', vp2, '->', (rt2.reshape(1,3) / rt2[2][0])[0][0:2])
	def project(p):
		proj = numpy.dot(numpy.dot(default_mtx, numpy.concatenate([r, tvec.reshape(3,1)], axis=1)), numpy.array(p).reshape(4,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]
	print('round trip 1', vp1, '->', project([1., 0., 0., 0.]))
	print('round trip 2', vp2, '->', project([0., 1., 0., 0.]))

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

	cv2.circle(bg, tuple(int(p) for p in vp1), 5, (0, 255, 0))
	cv2.circle(bg, tuple(int(p) for p in vp2), 5, (0, 255, 0))
	cv2.line(bg, tuple(int(p) for p in vp1), tuple(int(p) for p in vp2), (0,255,0), 1)

	#cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(0)

	tvec = numpy.array([-190., 80., 0.])
	def reverse_project(p):
		proj = numpy.dot(numpy.linalg.inv(numpy.dot(default_mtx, numpy.concatenate([numpy.delete(r, 2, 1), tvec.reshape(3,1)], axis=1))), numpy.array(p).reshape(3,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]

	bg = numpy.copy(color3)
	rq = []
	for quad in inlier_quads:
		quadpts = numpy.array([reverse_project([pt[0], pt[1], 1.]) for pt in quad])
		rq.append(quadpts)
		#print('reverse projected quad', quadpts)
		cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		cv2.drawContours(bg, [numpy.array([[dim * 100 for dim in pt] for pt in quadpts]).astype('int')], -1, (255, 0, 0), 2)
	#cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(0)

	# FIXME: Calculate the x and y deltas separately (ignoring lines' slant)
	# once it is known that all quads are oriented consistently!

	perimeter = 0
	for quad in rq:
		perimeter += cv2.arcLength((quad * 100.).astype('int'), closed=True)
	avg = perimeter / float(len(rq) * 400)
	# FIXME: Remove outliers and recalculate the average.
	print('avg-before', avg)

	# This could be framed as a mixed-integer linear programming (MILP)
	# problem, where each quad needs to be assigned to a integer grid
	# position. However, MILP tools are heavy-weight. A less robust
	# solution like this should always work adequately since there are
	# very few outliers remaining.

	# TODO: Calculate the most likely offset.
	# TODO: Then remove outliers and recalculate.


	zoom_in = numpy.array([[avg, 0., 0.], [0., avg, 0.], [0., 0., 1.]])
	scaled = numpy.dot(r, zoom_in)
	def reverse_project_scaled(p):
		proj = numpy.dot(numpy.linalg.inv(numpy.dot(default_mtx, numpy.concatenate([numpy.delete(scaled, 2, 1), tvec.reshape(3,1)], axis=1))), numpy.array(p).reshape(3,1))
		return (proj.reshape(1,3) / proj[2][0])[0][0:2]
	bg = numpy.copy(color3)
	rq = []
	for quad in inlier_quads:
		quadpts = numpy.array([reverse_project_scaled([pt[0], pt[1], 1.]) for pt in quad])
		rq.append(quadpts)
		cv2.drawContours(bg, [numpy.array(quad).astype('int')], -1, (0, 0, 255), 1)
		cv2.drawContours(bg, [numpy.array([[dim * 100 for dim in pt] for pt in quadpts]).astype('int')], -1, (255, 0, 0), 2)
	perimeter = 0
	for quad in rq:
		perimeter += cv2.arcLength((quad * 100.).astype('int'), closed=True)
	avg = perimeter / float(len(rq) * 400)
	print('avg-after', avg)
	cv2.imshow(WINNAME, bg)
	key = cv2.waitKey(0)


	return

	# TODO: Project all quads using the rotation matrix
	# TODO: Find the standard deviation of edge lengths and discard ouliers
	#

	#warped = cv2.warpPerspective(refimg, r, (color3.shape[1], color3.shape[0]))
	#cv2.imshow(WINNAME, warped)
	#key = cv2.waitKey(0)





	# FIXME: Is this true?
	invr = numpy.linalg.inv(r)
	homography = numpy.array([[invr[0][0], invr[0][1], 0.], [invr[1][0], invr[1][1], 0.], [invr[2][0], invr[2][1], 1.]])
	homography = numpy.array([[r[0][0], r[0][1], 0.], [r[1][0], r[1][1], 0.], [r[2][0], r[2][1], 1.]])
	sample_quads = inlier_quads
	sample_center_coords = [get_centroid(quad) for quad in sample_quads]

	# Move to a visible area of the image
	projected_centers = cv2.perspectiveTransform(
		numpy.array([sample_center_coords]).astype('float32'), homography)[0]
	group_center_x = sum(x for (x, y) in projected_centers) / float(len(projected_centers))
	group_center_y = sum(y for (x, y) in projected_centers) / float(len(projected_centers))
	#print('group_center', group_center_x, group_center_y)
	distance = (5. - group_center_x, 5. - group_center_y)
	translation = (distance[0] // 2. * 2, distance[1] // 2. * 2)
	#print('translation for show', translation)
	transM = numpy.array([[1., 0., translation[0]], [0., 1., translation[1]], [0., 0., 1.]])
	centeredM = numpy.dot(transM, homography)
	scaled = centeredM / centeredM[2][2]

	zoom_in = numpy.array([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
	#Min = numpy.dot(zoom_in, M)
	#invMin = numpy.dot(zoom_in, invM)
	zoomM = numpy.dot(zoom_in, scaled)
	reverseMdenorm = numpy.linalg.inv(zoomM)
	#reverseMdenorm = numpy.linalg.inv(scaled)
	reverseM = reverseMdenorm / reverseMdenorm[2][2]
	print('homography', homography)
	print('centered', centeredM)
	print('zoom', zoomM)
	print('reverse', reverseM)

	Mtrans = numpy.copy(homography)
	t1 = - Mtrans[0][2] / Mtrans[2][2]
	t2 = - Mtrans[1][2] / Mtrans[2][2]
	untrans = numpy.array([
		[1., 0., t1],
		[0., 1., t2],
		[0., 0., 1.],
	])
	#print('t', t1, t2)
	Muntrans = numpy.dot(untrans, Mtrans)
		#print('score', self.objective(sample_center_coords, true_center_coords, sample_quads, true_quads,
	#	numpy.array([cell for row in Muntrans for cell in row[:-1]])))
	warped = cv2.warpPerspective(refimg,
			reverseM,
			#numpy.dot(numpy.linalg.inv(zoom_in), homography),
			(color3.shape[1], color3.shape[0]))

	flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
	warpedmask = cv2.warpPerspective(flatmask, reverseM, (color3.shape[1], color3.shape[0]))
	maskidx = (warpedmask!=0)
	overlay = numpy.copy(color3)
	overlay[maskidx] = warped[maskidx]

	#print('quads', sample_quads)
	for quad in sample_quads:
		contours = [numpy.array([[point] for point in quad]).astype('int')]
		#cv2.drawContours(overlay, contours, -1, 255, cv2.FILLED)
		cv2.drawContours(overlay, contours, -1, 255, 2)
		#c = shapely.geometry.polygon.Polygon(quad).representative_point()
		c = [sum(p[dim] for p in quad)/4. for dim in [0, 1]]
		cv2.circle(overlay, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))

	cv2.imshow(WINNAME, overlay)
	#key = cv2.waitKey(0)


	projected = numpy.copy(color3)
	axis = numpy.float32([[0,0,0], [(pattern_size[0]+1)*100+1,0,0], [1,(pattern_size[1]+1)*100,0], [0,0,-100]]).reshape(-1,3)
	print('axis', axis)
	tvec = numpy.array([0., 0., 1.])
	dist = None
	(h, w) = color3.shape[0:2]
	default_mtx = numpy.array([[f, 0, w/2.], [0, f, h/2.], [0, 0, 1]]).astype('float32')
	invR = numpy.linalg.inv(r)
	axispt, j = cv2.projectPoints(axis, invR, tvec, default_mtx, dist)
	print('projected axis', axispt)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[1].ravel()), (0,0,255), 5)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[2].ravel()), (0,255,0), 5)
	cv2.line(projected, tuple(axispt[0].ravel()), tuple(axispt[3].ravel()), (255,0,0), 5)
	cv2.imshow(WINNAME, projected)
	#key = cv2.waitKey(0)



	reverseR = numpy.dot(numpy.linalg.inv(r), numpy.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
	print('revR', numpy.linalg.inv(r))
	#print('revR+rot', reverseR)
	myr = r
	tvec = numpy.array([-400., -300., 1.])

	#myr = reverseR
	#tvec = numpy.array([100., 200., 1.])

	#myr = invR
	#tvec = numpy.array([-400., -300., 1.])

	dist = None
	projectable_quads = numpy.array([[[pt[0], pt[1], 0.] for pt in quad] for quad in inlier_quads])
	bg = numpy.zeros(color3.shape)
	#bg = numpy.zeros([1000,2000,3])
	for pq in projectable_quads:
		quadpts, j = cv2.projectPoints(pq, myr, tvec, default_mtx, dist)
		quadpts = numpy.array([[int(dim) for dim in pt[0]] for pt in quadpts])
		#print('projected quad', quadpts)
		cv2.drawContours(bg, [quadpts], -1, (0, 0, 255), 1)
	cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(0)




	dist = None
	myr = r
	tvec = numpy.array([-1., -1., 1.])
	projectable_quads = numpy.array([[[x+dx, y+dy, 0.] for (dx,dy) in [(0,0),(1,0),(1,1),(0,1)]] for (x,y) in itertools.product(range(8), range(8))])
	bg = numpy.zeros(color3.shape)
	#bg = numpy.zeros([1000,2000,3])
	for pq in projectable_quads:
		quadpts, j = cv2.projectPoints(pq, myr, tvec, default_mtx, dist)
		quadpts = numpy.array([[int(dim) for dim in pt[0]] for pt in quadpts])
		#print('projected quad', quadpts)
		cv2.drawContours(bg, [quadpts], -1, (0, 0, 255), 1)
	print('r', myr)
	cv2.imshow(WINNAME, bg)
	#key = cv2.waitKey(0)

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
