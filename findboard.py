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
	contoursd = []
	contours = contoursd + contourse
	print('contours', len(contoursd), len(contourse), len(contours))

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
	print('good', len(goodd), len(goode), len(good))

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
	print('M', Mtrans)
	#t2 = (Mtrans[1][2] - Mtrans[1][0] * Mtrans[0][2] / Mtrans[0][0]) / (Mtrans[1][0] * Mtrans[0][1] / Mtrans[0][0] - Mtrans[1][1])
	#t1 = - (Mtrans[0][1] * t2 + Mtrans[0][2]) / Mtrans[0][0]
	t1 = - Mtrans[0][2] / Mtrans[2][2]
	t2 = - Mtrans[1][2] / Mtrans[2][2]
	untrans = numpy.array([
		[1., 0., t1],
		[0., 1., t2],
		[0., 0., 1.],
	])
	print('t', t1, t2)
	Muntrans = numpy.dot(untrans, invMout)
	print('M-untrans', Muntrans)

	# FIXME: Discard any RANSAC sample sets where the furthest points are more than 9 units apart.
	class ChessboardPerspectiveEstimator(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
		def __init__(self):
			self.homography_ = numpy.identity(3)
			# FIXME: This starts out with a known good value.
			self.homography_ = Muntrans
			self.untrans_homography_ = Muntrans
			#self.homography_ = invMout
			#self.untrans_homography_ = invMout
		def shape_homography(self, rotation):
			return numpy.concatenate((rotation.reshape(3, 2), [[0.], [0.], [1.]]), axis=1)
		def min_translation(self, true_centers, projected_centers):
			offsets = (projected_point - true_point for (projected_point, true_point) in zip(projected_centers, true_centers))
			# Shift every other row sideways, so all the like-colored squares line up.
			shifted = ((x + y % 2 // 1, y) for (x, y) in offsets)
			# We need the best translation that puts the points closest to true square points.
			# The translation in each dimension can be calculated independently.
			# It doesn't matter which square the points are near, so that means that modular arithmetic is needed.
			# The offsets need to be brought as close as possible to any integral values.
			# To do that, the points are translated to points on a unit circle and the average angle is computed.
			angles = ((x * math.pi, y * math.pi*2) for (x, y) in shifted)
			(anglesx, anglesy) = zip(*angles)
			biasanglex = math.atan2(
				sum(math.sin(angle) for angle in anglesx),
				sum(math.cos(angle) for angle in anglesx))
			biasangley = math.atan2(
				sum(math.sin(angle) for angle in anglesy),
				sum(math.cos(angle) for angle in anglesy))
			biasx = biasanglex / math.pi
			biasy = biasangley / (math.pi*2)
			translation = [biasx, biasy]
			return translation
		def objective(self, sample_centers, true_centers, sample_quads, true_quads, x):
			#print('obj', x)
			rotation = self.shape_homography(x)
			projected_centers = cv2.perspectiveTransform(
				numpy.array([sample_centers]).astype('float32'), rotation)[0]
			projected_quads = cv2.perspectiveTransform(
				numpy.array(sample_quads).astype('float32'), rotation)
			translation = self.min_translation(true_centers, projected_centers)
			translated_centers = (projected_center + translation for projected_center in projected_centers)
			# Shift every other row sideways, so all the like-colored squares line up.
			shifted_centers = ((x + y % 2 // 1, y) for (x, y) in translated_centers)
			mod_centers = ((x % 2, y % 1) for (x, y) in shifted_centers)
			quad_translations = (mod_center - projected_center
				for (projected_center, mod_center) in zip(projected_centers, mod_centers))
			rotated_quads = (self.rotate_quad(quad) for quad in sample_quads)
			translated_quads = ([(x + translationx, y + translationy) for (x, y) in quad]
				for ((translationx, translationy), quad) in zip(quad_translations, rotated_quads))

			offsets = ([true_dim - translated_dim for (true_dim, translated_dim) in zip(true_point, translated_point)]
				for (true_quad, translated_quad) in zip(true_quads, translated_quads)
				for (true_point, translated_point) in zip(true_quad, translated_quad))
			distance = sum(dim**2 for offset in offsets for dim in offset)

			#homography = numpy.append(x, [1]).reshape(3, 3)
			#projected_centers = cv2.perspectiveTransform(
			#	numpy.array([sample_centers]).astype('float32'), homography)[0]
			#offsets = (true_point - projected_point for (true_point, projected_point) in zip(true_centers, projected_centers))
			# Shift every other row sideways, so all the like-colored squares line up.

			#shifted = ((x + y % 2 // 1, y) for (x, y) in offsets)
			#mod = ((x % 2, y % 1) for (x, y) in shifted)

			#distance = sum(offset_x**2 + offset_y**2 for (offset_x, offset_y) in mod)
			#print('distance', distance)
			return distance
		def fit(self, X, y):
			print('fit', len(X))
			#print('FIT', X, y)
			scaled = self.untrans_homography_ / self.untrans_homography_[2][2]
			seed = numpy.array([cell for row in scaled for cell in row[:-1]])
			sample_quads = [grouper(quad, 2) for quad in X]
			sample_center_points = (shapely.geometry.polygon.Polygon(quad).representative_point() for quad in sample_quads)
			sample_center_coords = [(point.x, point.y) for point in sample_center_points]
			true_quads = [grouper(quad, 2) for quad in y]
			true_center_points = (shapely.geometry.polygon.Polygon(quad).representative_point() for quad in true_quads)
			true_center_coords = [(point.x, point.y) for point in true_center_points]
			#print('basinhopping...')
			#res = scipy.optimize.basinhopping(lambda x: self.objective(sample_center_coords, true_center_coords, sample_quads, true_quads, x), seed)
			#if not res.lowest_optimization_result.success:
			#	raise RuntimeError('solver failed: ' + res.lowest_optimization_result.message)
			#print('basinhopping done')
			#fitted = res.lowest_optimization_result.x

			# FIXME
			# The translation had already been determined deep inside the objective function,
			# but it was lost and needs to be recovered.
			#rotation = self.shape_homography(fitted)
			rotation = self.homography_
			self.untrans_homography_ = rotation

			projected_centers = cv2.perspectiveTransform(
				numpy.array([sample_center_coords]).astype('float32'), rotation)[0]
			translation = self.min_translation(true_center_coords, projected_centers)
			print('Translation', translation)
			transM = numpy.array([[1., 0., translation[0]], [0., 1., translation[1]], [0., 0., 1.]])
			#print('LAST')
			#self.show_homography(sample_center_coords, true_center_coords, sample_quads, true_quads, self.homography_)

			print('ROTATION')
			self.show_homography(sample_center_coords, true_center_coords, sample_quads, true_quads, rotation)

			self.homography_ = numpy.dot(transM, rotation)

			projected_quads = cv2.perspectiveTransform(
				numpy.array(sample_quads).astype('float32'), self.homography_)
			for (pq, tq) in zip(projected_quads, true_quads):
				rq = self.rotate_quad(pq)
				print('quad score', sum((td-rd)**2 for (rp, tp) in zip(rq, tq) for (rd, td) in zip(rp, tp)),
					rq, tq)

			#print('sample', sample_center_coords)
			#print('true', true_center_coords)
			#print('scaled', scaled)
			#self.homography_ = numpy.append(fitted, [1]).reshape(3, 3)
			print('TRANSLATED')
			# FIXME: In the objective function, the points should not be translated any more,
			# and this score should exactly equal ROTATION above!
			self.show_homography(sample_center_coords, true_center_coords, sample_quads, true_quads, self.homography_)

			corners_vis = numpy.copy(color2)
			for c in sample_center_coords:
				#corners_img	[(int(round(c[1])), int(round(c[0])))] = 255
				cv2.circle(corners_vis, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))
			#cv2.imshow(WINNAME, corners_vis)
			#key = cv2.waitKey(0)

			#warped = cv2.warpPerspective(refimg, M, (color3.shape[1], color3.shape[0]))
			#cv2.imshow(WINNAME, warped)
			#key = cv2.waitKey(0)

			return self
		def show_homography(self, sample_center_coords, true_center_coords, sample_quads, true_quads, homography):
			# Move to a visible area of the image
			sample_center_points = (shapely.geometry.polygon.Polygon(quad).representative_point() for quad in sample_quads)
			projected_centers = cv2.perspectiveTransform(
				numpy.array([sample_center_coords]).astype('float32'), homography)[0]
			group_center_x = sum(x for (x, y) in projected_centers) / float(len(projected_centers))
			group_center_y = sum(y for (x, y) in projected_centers) / float(len(projected_centers))
			#print('group_center', group_center_x, group_center_y)
			distance = (4. - group_center_x, 4. - group_center_y)
			translation = (distance[0] // 2. * 2, distance[1] // 2. * 2)
			#print('translation', translation)
			transM = numpy.array([[1., 0., translation[0]], [0., 1., translation[1]], [0., 0., 1.]])
			centeredM = numpy.dot(transM, homography)
			scaled = centeredM / centeredM[2][2]

			zoomM = numpy.dot(zoom_in, scaled)
			reverseM = numpy.linalg.inv(zoomM)

			print('score', self.objective(sample_center_coords, true_center_coords, sample_quads, true_quads,
				numpy.array([cell for row in scaled for cell in row[:-1]])))
			warped = cv2.warpPerspective(refimg,
					reverseM,
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

			cv2.imshow(WINNAME, overlay)
			key = cv2.waitKey(0)

			idealized = cv2.warpPerspective(color3,
					zoomM,
					(refimg.shape[1], refimg.shape[0]))
			projected_quads = cv2.perspectiveTransform(
				numpy.array(sample_quads).astype('float32'), zoomM)
			for quad in projected_quads:
				contours = [numpy.array([[point] for point in quad]).astype('int')]
				cv2.drawContours(idealized, contours, -1, 255, 2)

			cv2.imshow(WINNAME, idealized)
			key = cv2.waitKey(0)

		def score(self, X, y):
			print('score')
			s = super(ChessboardPerspectiveEstimator, self).score(X, y)
			#print('SCORE', X, y, s)
			return s
		def predict(self, X):
			print('predict')
			quads = [grouper(quad, 2) for quad in X]
			projected = cv2.perspectiveTransform(
				numpy.array(quads).astype('float32'), self.homography_)
			shifted = (self.rotate_quad(quad) for quad in projected)
			predicted = [[dim for corner in quad for dim in corner] for quad in shifted]
			#print('PREDICT', X, predicted)
			return predicted
		def rotate_quad(self, quad, debug=False):
			"""
			Transform the points by an integral distance and
			reorder the points so the upper-leftmost is first
			"""
			center = shapely.geometry.polygon.Polygon(quad).representative_point()
			# Shift every other row sideways, so all the like-colored squares line up.
			(shiftedx, shiftedy) = (
				# First find which group of two squares the center is in
				center.x // 2 * 2 +
				# Then decide whether to shift by one square
				((center.x + center.y % 2 // 1) +
				# Then find the right place within the group of two squares
				center.x) % 2,
				# The x coord already preserves the color, so the y coord doesn't need to shift.
				center.y)
			# Move to the two light and dark squares nearest the origin.
			(transformx, transformy) = (-(shiftedx // 2 * 2), -(shiftedy // 1))
			transformed = [(x + transformx, y + transformy) for (x, y) in quad]

			angles = (math.atan2(y - center.y, x - center.x) for (x, y) in quad)
			if debug:
				angles = list(angles)
			# It is assumed the points are already listed counter-clockwise and their sequence should be preserved.
			# If the homography inverted the orientation, then the error measurement will be high
			# and the homography will need to be discarded.
			# Each angle should be 90 degrees greater than the previous angle. All the angles will
			# have the same bias (angular distance from the desired orientation) if the quad is a perfect square.
			bias_angles = (angle + i * (math.pi/2.) for (i, angle) in enumerate(angles))
			if debug:
				bias_angles = list(bias_angles)
			average_bias_angle = math.atan2(
				sum(math.sin(angle) for angle in bias_angles),
				sum(math.cos(angle) for angle in bias_angles))
			# Reorder the points to minimize the bias.
			rotation = int(average_bias_angle / (math.pi/2.) + 2)
			#print('rotation', rotation)
			rotated = transformed[rotation:] + transformed[:rotation]
			if debug:
				print('--------------------------------')
				print('center', (center.x, center.y))
				print('shifted', (shiftedx, shiftedy))
				print('transform', (transformx, transformy))
				print('new center', (center.x + transformx, center.y + transformy))
				print('offsets', [(x - center.x, y - center.y) for (x, y) in quad])
				print('raw angles', [a / (math.pi/2.) for a in angles])
				print('bias angles', [a / (math.pi/2.) for a in bias_angles])
				print('average bias angle', average_bias_angle / (math.pi/2.))
				print('rotation', rotation)
				print('rotated from', transformed)
				print('rotated to', rotated)
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
		residual_threshold=5,
	)
	# RANSACRegressor expects the input to be an array of points.
	# This target data is an array of quads instead, where each quad
	# contains 4 points. The translation is done by passing all 4 2-D
	# points as if they were a single 8-dimensional point.
	target_pts = [[dim for corner in quad for dim in corner] for quad in quads]
	dark_square = (0., 0., 0., 1., 1., 1., 1., 0.)
	light_square = (1., 0., 1., 1., 2., 1., 2., 0.)
	training_pts = [(light_square if i < len(goodd) else dark_square) for i in range(len(quads))]
	regressor.fit(target_pts, training_pts)
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
	return (numpy.mean([p[0] for p in points]), numpy.mean([p[1] for p in points]))


if __name__ == "__main__":
	main()
