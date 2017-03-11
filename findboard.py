#!/usr/bin/env python

import colorsys
import cv2
import itertools
import math
import numpy
import random
import scipy.optimize
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
	#[webcam.read() for i in range(500)]

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
	thresh = cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,0)
	#cv2.imshow(WINNAME, thresh)
	#key = cv2.waitKey(0)

	# For finding dark squares
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,5))
	dilated = cv2.dilate(thresh, kernel)
	# For finding light squares
	kernel = numpy.ones((3,5),numpy.uint8)
	eroded = cv2.erode(thresh, kernel)
	#cv2.imshow(WINNAME, dilated)
	#key = cv2.waitKey(0)
	#cv2.imshow(WINNAME, eroded)
	#key = cv2.waitKey(0)


	im2, contoursd, hierarchy = cv2.findContours(dilated,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	ime2, contourse, hierarchy = cv2.findContours(eroded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = contoursd + contourse
	#print('contours', len(contoursd), len(contourse), len(contours))

	good = []
	goodd = []
	goode = []
	for (idx, contour) in enumerate(contours):
		perimeter = cv2.arcLength(contour, True)
		if len(contour) > 4:
			approx = cv2.approxPolyDP(contour, perimeter/20, True)
		elif len(contour) == 4:
			approx = contour
		else:
			continue

		#if len(approx) > 4:
		#	approx = cv2.approxPolyDP(approx, 5, True)
		if len(approx) != 4:
			continue
			pass
		if not cv2.isContourConvex(approx):
			continue
		#if cv2.arcLength(approx,True) < 50:
		#	continue
		if cv2.contourArea(approx) < 40:
			continue

		# FIXME: Remove bookshelf contours
		if approx[0][0][1] < 300:
			continue

		# Dilate the contour
		bg = numpy.zeros((color2.shape[0], color2.shape[1]), dtype=numpy.uint8)
		cv2.drawContours(bg, [approx], -1, 255, cv2.FILLED)
		#kernel_lg = numpy.ones((20,30),numpy.uint8)
		dilatedApprox = cv2.dilate(bg, kernel)
		# TODO: This is messy. Use a mathematical translation of the contour
		# to find the contour that barely bounds this contour with a
		# kernel mask at each corner. This can be done with cv2.convexHull
		imda, contoursa, hierarchy = cv2.findContours(dilatedApprox,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		if len(contoursa) != 1:
			print('bad contour count', contoursa)
		contoursa1 = contoursa[0]
		if len(contoursa1) > 4:
			#print('repeating', len(contoursa1))
			contoursa1 = cv2.approxPolyDP(contoursa1, perimeter/20, True)
		if len(contoursa1) != 4:
			contoursa1 = approx

		# FIXME: Find sub-pixel position of these corners.

		good.append(contoursa1)
		if idx < len(contoursd):
			goodd.append(contoursa1)
		else:
			goode.append(contoursa1)
	#print('good', len(goodd), len(goode), len(good))

	contcorners = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
	contlines = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
	contlinesd = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
	contlinese = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
	for (idx, contour) in reversed(list(enumerate(good))):
		#color = [a*256 for a in colorsys.hsv_to_rgb(random.random(), 0.75, 1)]
		color = (255, 0, 0)
		#cv2.drawContours(color1, [contour], -1, color, cv2.FILLED)
		cv2.drawContours(color1, [contour], -1, color, 2)
		cv2.drawContours(contlines, [contour], -1, 255, 1)
		if idx < len(goodd):
			cv2.drawContours(contlinesd, [contour], -1, 255, 1)
		else:
			cv2.drawContours(contlinese, [contour], -1, 255, 1)
		for cr in contour:
			if contcorners[(cr[0][1], cr[0][0])]:
				#print('clobbered corner')
				pass
			contcorners[(cr[0][1], cr[0][0])] = 255
		#print('area', cv2.contourArea(contour))
		#cv2.imshow(WINNAME, color1)
		#key = cv2.waitKey(0)

	#cv2.imshow(WINNAME, contlinesd)
	#key = cv2.waitKey(0)
	#cv2.imshow(WINNAME, contlinese)
	#key = cv2.waitKey(0)

	#cv2.imshow(WINNAME, contlines)
	#key = cv2.waitKey(0)






	# FIXME: Temp code to find chessboard corners

	#cv2.imshow(WINNAME, img1)
	#key = cv2.waitKey(0)
	(found, corners) = cv2.findChessboardCorners(img1, pattern_size)
	#print('chessboard', found, corners)
	clus_corners = corners = [c[0] for c in corners]

	def get_odd_corners(c):
		#return [c[1], c[7], c[15], c[21]]
		return [c[i] for i in xrange(1, len(c), 6)]
	def get_even_corners(c):
		#return [c[8], c[14], c[22], c[28]]
		return [c[i] for i in xrange(2, len(c), 6)]
	def get_corners(c):
		#return get_odd_corners(c) + get_even_corners(c)
		return c
	corners_img = numpy.zeros(img1.shape, numpy.uint8)
	corners_vis = numpy.copy(color2)
	for c in get_corners(clus_corners):
		#corners_img[(int(round(c[1])), int(round(c[0])))] = 255
		cv2.circle(corners_vis, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))
		pass
	#corners_img[corners]= 255
	#color2[corners]= 255




	for c in ref_corners:
		cv2.circle(refimg, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))
		pass
	#cv2.imshow(WINNAME, refimg)
	#key = cv2.waitKey(0)

	src_pts = numpy.float32([ pt for pt in get_corners(reversed(ref_corners)) ]).reshape(-1,1,2)
	dst_pts = numpy.float32([ pt for pt in get_corners(clus_corners) ]).reshape(-1,1,2)


	# Calibrate the camera.
	obj_pts = numpy.array([[(p[0][0], p[0][1], 0) for p in src_pts]]).astype('float32')
	#print(obj_pts)
	img_pts = numpy.array([[p[0] for p in dst_pts]])
	#print(img_pts)
	h, w = color2.shape[:2]
	#print('mtx', mtx)
	#print('newmtx', newcameramtx)

	f = max(w, h)
	default_mtx = numpy.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]).astype('float32')

	#ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist)
	#ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, None)
	ret, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, default_mtx, None)

	mtx = default_mtx
	dist = None

	color3 = numpy.copy(color2)

	# TODO: Once the board is found, use MSER to track it

	# FIXME
	# Use sklearn.linear_model.RANSACRegressor to determine outliers and
	# scipy.optimize with method=lm (Levenberg-Marquardt algorithm)
	# to fit the corners to a perspective.

	M, hmask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

	class ChessboardProjectiveTransform(skimage.transform.ProjectiveTransform):
		pass

	points = [p for p in contour for contour in contours]
	ideal_points = [numpy.array([(0, 0)]) for p in points]
	#print(points)
	#print(ideal_points)

	src_input = numpy.array([p[0] for p in src_pts])
	dst_input = numpy.array([p[0] for p in dst_pts])
	#print(src_input)
	#print(dst_input)
	model, inliers = skimage.measure.ransac(
		#(numpy.array(ideal_points), numpy.array(points)),
		(src_input, dst_input),
		ChessboardProjectiveTransform,
		min_samples=10, residual_threshold=5, max_trials=500)



	a = math.pi / 2.
	rot90 = numpy.array([[math.cos(a), math.sin(a), 0.], [-math.sin(a), math.cos(a), 0.], [0., 0., 1.]])
	M = numpy.dot(M, rot90)

	#print('INLIERS', inliers)

	#M = model.params
	#ret, rvecs, tvecs, nor = cv2.decomposeHomographyMat(M, mtx)

	#print('RVECS', rvecs)
	#print('RVEC_OLD', rvec)
	#print('TVECS', tvecs)
	#print('TVEC_OLD', tvec)
	#(rvec, tvec) = (rvecs[0][0], tvecs[0])

	(rotationMatrix, jacobian) = cv2.Rodrigues(rvec)
	#tempMat2 = rotationMatrix.inv() * tvec;
	#tempMat = rotationMatrix.inv() * mtx.inv() * uvPoint;
	#print("rotationMatrix", rotationMatrix)
	#print("jacobian", jacobian)
	#print("mtx", mtx)
	#print("rvec", rvec)
	#print("tvec", tvec)
	invRotationMtx = numpy.linalg.inv(rotationMatrix)
	invMtx = numpy.linalg.inv(mtx)

	#points = numpy.array([[good[0][0][0].astype('float32')]])
	#print('point', points[0][0])
	#print('M', M)
	invM = numpy.linalg.inv(M)
	#print('invM', invM)
	#print('projected', invRotationMtx * (invMtx * point - tvec))
	#proj = cv2.perspectiveTransform(points, M)
	#print('projected', proj[0][0])
	#ret = cv2.perspectiveTransform(proj, invM)
	#print('ret', ret[0][0])


	#M[2][0] *= 10.
	#M[2][1] *= 1.
	#M[0][0] *= 10.
	#M[0][2] += 10000.
	#print('M', M)
	for shift in xrange(-10000, 10000, 1000):
		break
		shiftM = numpy.copy(M)
		shiftM[0][2] += shift
		print('M', shift, shiftM)
		warpedtest = cv2.warpPerspective(refimg, shiftM, (color2.shape[1], color2.shape[0]))
		cv2.imshow(WINNAME, warpedtest)
		key = cv2.waitKey(0)

	#invM[2][0] *= 10.
	#invM[2][1] *= 1.
	a = 0.75
	#invM = numpy.dot(numpy.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]), invM)
	#invM[0][1] *= 1.5
	#invM[1][0] /= 1.5
	last = 0
	for shift in xrange(-500, 500, 50):
		break
		shiftM = numpy.copy(invM)
		shiftM = numpy.dot(numpy.array([[1., 0., float(shift)], [0., 1., 0.], [0., 0., 1.]]), invM)
		#print('M', shift, shiftM)

		projected = cv2.perspectiveTransform(
			numpy.array([src_pts[0]]).astype('float32'), shiftM)[0]
		print('projected', projected[0][0], projected[0][0] - last)
		last = projected[0][0]

		warpedtest = cv2.warpPerspective(color2, shiftM, (refimg.shape[1], refimg.shape[0]))
		cv2.imshow(WINNAME, warpedtest)
		key = cv2.waitKey(250)

	zoom_out = numpy.array([[0.01, 0., 0.], [0., 0.01, 0.], [0., 0., 1.]])
	Mout = numpy.dot(zoom_out, M)
	invMout = numpy.dot(zoom_out, invM)
	warpedtest = cv2.warpPerspective(refimg, M, (color2.shape[1], color2.shape[0]))
	#cv2.imshow(WINNAME, warpedtest)
	#key = cv2.waitKey(0)

	zoom_in = numpy.array([[100., 0., 0.], [0., 100., 0.], [0., 0., 1.]])
	Min = numpy.dot(zoom_in, M)
	invMin = numpy.dot(zoom_in, invM)

	warpedtest = cv2.warpPerspective(color2, invMin, (refimg.shape[1], refimg.shape[0]))
	warpedtest = cv2.warpPerspective(color2, invM, (8, 8))
	#cv2.imshow(WINNAME, warpedtest)
	#key = cv2.waitKey(0)


	#ret, rvecs, tvecs, nor = cv2.decomposeHomographyMat(M, mtx)
	#print('RVECS', rvecs)
	#print('TVECS', tvecs)
	#print('NOR', nor)

	Mtrans = numpy.copy(invMout)
	#Mtrans = Mtrans / Mtrans[2][2]
	#print('M', Mtrans)
	#t2 = (Mtrans[1][2] - Mtrans[1][0] * Mtrans[0][2] / Mtrans[0][0]) / (Mtrans[1][0] * Mtrans[0][1] / Mtrans[0][0] - Mtrans[1][1])
	#t1 = - (Mtrans[0][1] * t2 + Mtrans[0][2]) / Mtrans[0][0]
	t1 = - Mtrans[0][2] / Mtrans[2][2]
	t2 = - Mtrans[1][2] / Mtrans[2][2]
	untrans = numpy.array([
		[1., 0., t1],
		[0., 1., t2],
		[0., 0., 1.],
	])
	#print('t', t1, t2)
	Muntrans = numpy.dot(untrans, Mtrans)
	#print('M-untrans', Muntrans)

	# FIXME: Discard any RANSAC sample sets where the furthest points are more than 9 units apart.
	class ChessboardPerspectiveEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
		def __init__(self):
			self.seed = [color3.shape[1]/2., color3.shape[0]/2., float(color3.shape[1]), color3.shape[0]/2.]
			self.vps = None
		def segment_alignment_error(self, segment, vp, debug=False):
			oriented = sorted(segment, key=lambda point: numpy.linalg.norm(point - vp), reverse=True)
			(closer, further) = oriented
			direction = closer - further
			vanisher = vp - further
			unit_vanisher = vanisher / numpy.linalg.norm(vanisher)
			corrected = numpy.dot(direction, unit_vanisher) * unit_vanisher
			distance = numpy.linalg.norm(corrected - direction)
			if debug:
				print('projecting', corrected, direction, vanisher)
				print('correction', distance, list(closer), '=>', list(further + corrected))
			return distance**2
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
			unit_horizon = horizon / numpy.linalg.norm(horizon)
			distance = 0.

			projected_quad = []
			for point in quad:
				# Check which side of the horizon each point is on.
				if numpy.linalg.det([horizon, point - vp1]) < 0:
					# Move any points above the horizon directly onto the horizon
					projection = numpy.dot(point - vp1, unit_horizon)
					projected_point = vp1 + projection * unit_horizon
					distance += numpy.linalg.norm(projected_point - point)**2
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

			#print('distance', distance)

			#working = numpy.copy(color3)
			#left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
			#right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
			#cv2.drawContours(working, numpy.array(sample_quads), -1, 255, 2)
			#cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 0, 255))
			#cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			#cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 0, 255))
			#cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			#cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			#cv2.imshow(WINNAME, working)
			#key = cv2.waitKey(0 if wait else 1)

			return distance
		def fit(self, X, y):
			sample_quads = [grouper(quad, 2) for quad in X]

			print('optimizing...')
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
			res = scipy.optimize.minimize(lambda x: optimizer(x), self.seed, method='L-BFGS-B')
			if not res.success:
				raise RuntimeError('solver failed: ' + res.message)
			fitted = res.x
			self.vps = numpy.array(grouper(fitted, 2))

			print('optimization done')

			(vp1, vp2) = (numpy.array(vp) for vp in grouper(fitted, 2))
			working = numpy.copy(color3)
			left_center = (color3.shape[1]//2 - 50, color3.shape[0]//2 - 25)
			right_center = (color3.shape[1]//2 + 50, color3.shape[0]//2 + 25)
			cv2.drawContours(working, numpy.array(sample_quads), -1, 255, 2)
			cv2.circle(working, tuple(int(p) for p in vp1), 5, (0, 0, 255))
			cv2.line(working, left_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			cv2.line(working, right_center, tuple(int(p) for p in vp1), (0,0,255), 2)
			cv2.circle(working, tuple(int(p) for p in vp2), 5, (0, 0, 255))
			cv2.line(working, left_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			cv2.line(working, right_center, tuple(int(p) for p in vp2), (0,0,255), 2)
			cv2.imshow(WINNAME, working)
			key = cv2.waitKey(1)

			#print('comparison', self.objective(sample_quads, [360., 200., 6000., 240.], True))

			# FIXME
			# The translation had already been determined deep inside the objective function,
			# but it was lost and needs to be recovered.

			return self
		def score(self, X, y):
			print('score')
			s = super(ChessboardPerspectiveEstimator, self).score(X, y)
			#print('SCORE', X, y, s)
			return s
		def predict(self, X):
			print('predict')
			quads = [grouper(quad, 2) for quad in X]
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

	# Change the shape to a list of quads.
	# Currently each square is 100 pixels per size, so normalize it down to unit squares.
	#quads = [[[dim / 100. for dim in corner[0]] for corner in quad] for quad in good]
	quads = [[[dim for dim in corner[0]] for corner in quad] for quad in good]
	regressor = sklearn.linear_model.RANSACRegressor(
		base_estimator=ChessboardPerspectiveEstimator(),
		min_samples=3,
		residual_metric=lambda dy: numpy.sum(dy**2, axis=1),
		#residual_threshold=1/8.,
		residual_threshold=10.,
	)
	# RANSACRegressor expects the input to be an array of points.
	# This target data is an array of quads instead, where each quad
	# contains 4 points. The translation is done by passing all 4 2-D
	# points as if they were a single 8-dimensional point.
	target_pts = [[dim for corner in quad for dim in corner] for quad in quads]
	dark_square = (0., 0., 0., 1., 1., 1., 1., 0.)
	light_square = (1., 0., 1., 1., 2., 1., 2., 0.)
	training_pts = [(0,) for i in range(len(quads))]
	regressor.fit(target_pts, training_pts)
	# FIXME
	M = numpy.linalg.inv(numpy.dot(zoom_in, regressor.estimator_.homography_))



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


if __name__ == "__main__":
	main()
