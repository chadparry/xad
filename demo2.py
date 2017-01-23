#!/usr/bin/env python

import colorsys
import cv2
import itertools
import math
import numpy
import random
import sklearn.cluster
import sklearn.datasets.samples_generator


WINNAME = 'Chess Transcription'



def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = numpy.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = numpy.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = numpy.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out


def main():

	#webcam = cv2.VideoCapture(0)
	webcam = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')
	#webcam = cv2.VideoCapture('idaho.webm')
	if not webcam.isOpened():
		raise RuntimeError('Failed to open camera')

	#[webcam.read() for i in range(10)]
	#[webcam.read() for i in range(500)]
	(retval, color2) = webcam.read()

	img2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

	refimgbw = cv2.imread('chessboard.png',0)
	refimg = cv2.cvtColor(refimgbw, cv2.COLOR_GRAY2BGR)
	pattern_size = (7, 7)
	(found, ref_chess) = cv2.findChessboardCorners(refimg, pattern_size)
	ref_corners = [c[0] for c in ref_chess]
	#print('chessboard corners', ref_corners)

	cv2.namedWindow(WINNAME)

	while True:
		(retval, color1) = webcam.read()
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
		#contours = contourse
		contours = contoursd + contourse

		good = []
		for contour in contours:
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

			# Dilate the contour
			bg = numpy.zeros((color2.shape[0], color2.shape[1]), dtype=numpy.uint8)
			cv2.drawContours(bg, [approx], -1, 255, cv2.FILLED)
			#kernel_lg = numpy.ones((20,30),numpy.uint8)
			dilatedApprox = cv2.dilate(bg, kernel)
			# TODO: This is messy. Use a mathematical translation of the contour
			# to find the contour that barely bounds this contour with a
			# kernel mask at each corner.
			imda, contoursa, hierarchy = cv2.findContours(dilatedApprox,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
			if len(contoursa) != 1:
				print('bad contour count', contoursa)
			contoursa1 = contoursa[0]
			if len(contoursa1) > 4:
				#print('repeating', len(contoursa1))
				contoursa1 = cv2.approxPolyDP(contoursa1, perimeter/20, True)
			if len(contoursa1) != 4:
				contoursa1 = approx

			good.append(contoursa1)

		contcorners = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
		contlines = numpy.zeros((color2.shape[0], color2.shape[1]), numpy.uint8)
		for contour in reversed(good):
			#color = [a*256 for a in colorsys.hsv_to_rgb(random.random(), 0.75, 1)]
			color = (255, 0, 0)
			#cv2.drawContours(color1, [contour], -1, color, cv2.FILLED)
			cv2.drawContours(color1, [contour], -1, color, 2)
			cv2.drawContours(contlines, [contour], -1, 255, 1)
			for cr in contour:
				if contcorners[(cr[0][1], cr[0][0])]:
					#print('clobbered corner')
					pass
				contcorners[(cr[0][1], cr[0][0])] = 255
			#print('area', cv2.contourArea(contour))
			#cv2.imshow(WINNAME, color1)
			#key = cv2.waitKey(0)

		#cv2.imshow(WINNAME, contlines)
		#key = cv2.waitKey(0)

		src = numpy.array([(3, 3), (4, 3), (4, 4), (3, 4)], dtype=numpy.float32)
		allboards = numpy.copy(color2)
		ALPHA_WEIGHT = 0.01
		matrices = []
		for contour in reversed(good):
			c = numpy.array([pt[0] for pt in contour], dtype=numpy.float32)
			if len(c) != 4:
				continue
			M = cv2.getPerspectiveTransform(src, c)
			matrices.append([M[x][y] for x in range(3) for y in range(3)])
			#matrices.append(M)
			#print('matrix', M)


			#warped = cv2.warpPerspective(refimg, M, (color2.shape[1], color2.shape[0]))
			#flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
			#warpedmask = cv2.warpPerspective(flatmask, M, (color2.shape[1], color2.shape[0]))

			#maskidx = (warpedmask!=0)
			#overlay = numpy.copy(allboards)
			#overlay[maskidx] = warped[maskidx]

			#allboards = cv2.addWeighted(overlay, ALPHA_WEIGHT, allboards, 1 - ALPHA_WEIGHT, 0)

			#color = (0, 255, 255)
			#cv2.drawContours(overlay, [contour], -1, color, cv2.FILLED)

			#cv2.imshow(WINNAME, overlay)
			#key = cv2.waitKey(0)


		#centers = [[1, 1], [-1, -1], [1, -1]]
		#X, labels_true = sklearn.datasets.samples_generator.make_blobs(n_samples=20, centers=centers, cluster_std=0.4, random_state=0)
		#print('X', X)
		matrices = numpy.array(matrices)
		#print('matrices', matrices)

		SHEAR_FACTOR = 1
		ROTATION_FACTOR = 1
		SCALING_FACTOR = 1
		def decompose_matrix(m):
			p = math.sqrt(m[0]**2 + m[1]**2)
			r = (m[0]*m[4] - m[1]*m[3]) / p
			q = (m[0]*m[3] + m[2]*m[4]) / (m[0]*m[4] - m[1]*m[3])
			a = math.atan2(m[1], m[0])
			shear = math.sqrt(p**2 + r**2) * SHEAR_FACTOR
			rotation = math.sqrt(q**2 + 2) * ROTATION_FACTOR
			scaling = 1 * SCALING_FACTOR
			#print('decomposed', m, [shear, rotation, scaling])
			return [shear, rotation, scaling]

		def pers_dist(a, b):
			#return random.random()*1000
			return math.sqrt(sum((ap - bp)**2 for (ap, bp) in
				zip(decompose_matrix(a), decompose_matrix(b))))

		## Use k-NN or DBSCAN on every perspective matrix to find the most likely matrix
		#scanner = sklearn.cluster.DBSCAN(eps=50, min_samples=4, metric=pers_dist)
		##db = scanner.fit(X)
		#db = scanner.fit(matrices)
		##core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
		##core_samples_mask[db.core_sample_indices_] = True
		#print('kept', len(db.core_sample_indices_), len(matrices))
		
		for cidx, contour in enumerate(reversed(good)):
			break

			if cidx not in db.core_sample_indices_:
				continue


			c = numpy.array([pt[0] for pt in contour], dtype=numpy.float32)
			M = cv2.getPerspectiveTransform(src, c)
			#matrices.append([M[x][y] for x in range(3) for y in range(3)])
			#matrices.append(M)
			#print('matrix', M)


			warped = cv2.warpPerspective(refimg, M, (color2.shape[1], color2.shape[0]))
			flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
			warpedmask = cv2.warpPerspective(flatmask, M, (color2.shape[1], color2.shape[0]))

			maskidx = (warpedmask!=0)
			overlay = numpy.copy(allboards)
			overlay[maskidx] = warped[maskidx]

			allboards = cv2.addWeighted(overlay, ALPHA_WEIGHT, allboards, 1 - ALPHA_WEIGHT, 0)

			color = (0, 255, 255)
			cv2.drawContours(overlay, [contour], -1, color, cv2.FILLED)

			cv2.imshow(WINNAME, overlay)
			key = cv2.waitKey(0)


		#cv2.imshow(WINNAME, allboards)
		#key = cv2.waitKey(0)

		lines = cv2.HoughLines(contlines, 2, numpy.pi / 180, 200)
		#print('lines', lines)
		for lin in lines:
		    #print('line', lin[0])
		    (rho, theta) = lin[0]
		    #print('rho/theta', rho, theta)
		    a = numpy.cos(theta)
		    b = numpy.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))
		    #cv2.line(color2,(x1,y1),(x2,y2),(255,0,0),2)

		#cv2.imshow(WINNAME, color1)
		#key = cv2.waitKey(0)

		#cv2.imshow(WINNAME, color2)
		#key = cv2.waitKey(0)

		#cv2.imshow(WINNAME, contlines)
		#key = cv2.waitKey(0)
		#f = numpy.fft.fft2(contlines)
		#dft = cv2.dft(numpy.float32(contlines),flags = cv2.DFT_COMPLEX_OUTPUT)
		#dft_shift = numpy.fft.fftshift(dft)
		#dft_re, dft_im = cv2.split(dft_shift)
		#cv2.imshow(WINNAME, dft_re)
		#key = cv2.waitKey(0)
		#cv2.imshow(WINNAME, dft_im)
		#key = cv2.waitKey(0)

		#return

		# TODO: Once the board is found, use MSER to track it

		break

	[webcam.read() for i in range(300)]
	(retval, color3) = webcam.read()

	#return

	#img2 = img1
	#img1 = cv2.imread('/home/cparry/Pictures/chessboard.png',0)


	#dst = cv2.cornerHarris(img2,6,3,0.04)
	#corners = dst > 0.01 * dst.max()

	#print('max', dst.max())
	#print('dst', len(dst>0.01*dst.max()), len(dst))


	#corner_pt = []
	#for x in range(corners.shape[0]):
	#	for y in range(corners.shape[1]):
	#		if corners[x][y]:
	#			cv2.circle(color2, (y, x), 3, (0, 255, 0))
	#			corner_pt.append((y, x))
	#clus_corners = cluster_points(corner_pt)
	#color2[corners]=[0,0,255]

	pattern_size = (7, 7)
	(found, corners) = cv2.findChessboardCorners(img1, pattern_size)
	#print('chessboard', found, corners)
	clus_corners = corners = [c[0] for c in corners]


	def get_odd_corners(c):
		return [c[1], c[7], c[15], c[21]]
	def get_even_corners(c):
		return [c[8], c[14], c[22], c[28]]
	def get_corners(c):
		return get_odd_corners(c) + get_even_corners(c)
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

	src_pts = numpy.float32([ pt for pt in get_corners(ref_corners) ]).reshape(-1,1,2)
	dst_pts = numpy.float32([ pt for pt in get_corners(clus_corners) ]).reshape(-1,1,2)
	src_corners = []
	dst_corners = []
	for sc, dc in itertools.chain(
			itertools.product(get_odd_corners(ref_corners), get_odd_corners(clus_corners)),
			itertools.product(get_even_corners(ref_corners), get_even_corners(clus_corners)),
		):
		src_corners.append(sc)
		dst_corners.append(dc)
	matches = []
	for sc, dc in itertools.chain(
			#itertools.product(get_odd_corners(ref_corners), get_odd_corners(clus_corners)),
			#itertools.product(get_even_corners(ref_corners), get_even_corners(clus_corners)),
			itertools.product(range(len(get_odd_corners(ref_corners))), range(len(get_odd_corners(clus_corners)))),
			itertools.product(range(len(get_odd_corners(ref_corners)), len(get_odd_corners(ref_corners))+len(get_even_corners(ref_corners))), range(len(get_odd_corners(clus_corners)),len(get_odd_corners(clus_corners))+len(get_even_corners(clus_corners)))),
		):
		matches.append(cv2.DMatch(sc, dc, 1))
	#src_pts = numpy.float32(src_corners).reshape(-1,1,2)
	#dst_pts = numpy.float32(dst_corners).reshape(-1,1,2)

	cv2.imshow(WINNAME, corners_vis)
	key = cv2.waitKey(0)
	def mkpt(c):
		return cv2.KeyPoint(c[0], c[1], 5)
	refimgbw = cv2.cvtColor(refimg, cv2.COLOR_BGR2GRAY)
	bw2 = cv2.cvtColor(color2, cv2.COLOR_BGR2GRAY)

	#matches_vis = drawMatches(
	#	refimgbw,[mkpt(c) for c in get_corners(ref_corners)],
	#	bw2,[mkpt(c) for c in get_corners(clus_corners) ],matches)


	# FIXME
	# Use sklearn.linear_model.RANSACRegressor to determine outliers and
	# scipy.optimize with method=lm (Levenberg-Marquardt algorithm)
	# to fit the corners to a perspective.
	M, hmask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	hmatchesMask = hmask.ravel().tolist()

	#h,w = img1.shape
	#pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	#dst = cv2.perspectiveTransform(pts,M)

	warped = cv2.warpPerspective(refimg, M, (color3.shape[1], color3.shape[0]))
	#color3[0:warped.shape[0], 0:warped.shape[1]] = warped


	flatmask = numpy.full((refimg.shape[0], refimg.shape[1]), 255, dtype=numpy.uint8)
	flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
	warpedmask = cv2.warpPerspective(flatmask, M, (color3.shape[1], color3.shape[0]))

	maskidx = (warpedmask!=0)
	overlay = numpy.copy(color3)
	overlay[maskidx] = warped[maskidx]
	#warpedpartial = cv2.bitwise_and(warped, warpedmask)
	#invmask = cv2.bitwise_not(warpedmask)
	#color2partial = cv2.bitwise_and(color2, invmask)
	#overlay = cv2.bitwise_or(color2partial, warpedpartial)

	cv2.imshow(WINNAME, overlay)
	key = cv2.waitKey(0)

	idealized = cv2.warpPerspective(color3, M, (refimg.shape[1], refimg.shape[0]),
		flags=cv2.WARP_INVERSE_MAP)
	cv2.imshow(WINNAME, idealized)
	key = cv2.waitKey(0)

	return

	#######################################################

	lines = cv2.HoughLines(corners_img, 1, numpy.pi / 180, 5)[0]
	#print('lines', lines)
	#for x1,y1,x2,y2 in lines:        
	#	cv2.line(color2,(x1,y1),(x2,y2),(255,0,0),2)
	for rho,theta in lines:        
	    a = numpy.cos(theta)
	    b = numpy.sin(theta)
	    x0 = a*rho
	    y0 = b*rho
	    x1 = int(x0 + 1000*(-b))
	    y1 = int(y0 + 1000*(a))
	    x2 = int(x0 - 1000*(-b))
	    y2 = int(y0 - 1000*(a))
	    #cv2.line(color2,(x1,y1),(x2,y2),(255,0,0),2)


	for c in clus_corners:
		cv2.circle(color1, (int(round(c[0])), int(round(c[1]))), 3, (0, 255, 0))


	cv2.imshow(WINNAME, color1)
	key = cv2.waitKey(0)
	return


	MIN_MATCH_COUNT = 10

	# Initiate SIFT detector
	sift = cv2.SIFT()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

	pattern_size = (7, 7)
	(found1, corners1) = cv2.findChessboardCorners(img1, pattern_size)
	(found2, corners2) = cv2.findChessboardCorners(img2, pattern_size)
	kp1 = [cv2.KeyPoint(c[0][0], c[0][1], 5) for c in corners1]
	kp2 = [cv2.KeyPoint(c[0][0], c[0][1], 5) for c in corners2]

	(kp1, des1) = sift.compute(img1, kp1)
	(kp2, des2) = sift.compute(img2, kp2)
	#print('kp1', [(k.pt, k.size) for k in kp1])
	#print('kp2', [(k.pt, k.size) for k in kp2])


	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)

	neighbors = pattern_size[0] * pattern_size[1]

	# FIXME: A matcher looks at local features, which doesn't work!
	# I want matching to be done based on relative coordinates.
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=neighbors)

	#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	#matches = bf.knnMatch(des1, des2, k=2)

	# store all the good matches as per Lowe's ratio test.
	good = []
	#for m,n in matches:
	#    if m.distance < 0.7*n.distance:
	#        good.append(m)

	# Brute force every match from one interior corner to another
	for ms in matches:
		for m in ms:
			good.append(m)


	if len(good)>MIN_MATCH_COUNT:
	    src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	    matchesMask = mask.ravel().tolist()

	    h,w = img1.shape
	    pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    dst = cv2.perspectiveTransform(pts,M)

	    #img2 = cv2.polylines(img2,[numpy.int32(dst)],True,255,3, cv2.CV_AA)
	    cv2.polylines(img2,[numpy.int32(dst)],True,255,3, cv2.CV_AA)
	    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	                   singlePointColor = None,
	                   matchesMask = matchesMask, # draw only inliers
	                   flags = 2)

	    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
	    img3 = drawMatches(img1,kp1,img2,kp2,good)

	    warped = cv2.warpPerspective(img1, M, (800,800))
	    cv2.imshow('Warped', warped)
	    cv2.waitKey(0)
	    cv2.destroyWindow('Warped')

	else:
	    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	    matchesMask = None
	    cv2.drawChessboardCorners(img1, pattern_size, corners1, False)
	    cv2.drawChessboardCorners(img2, pattern_size, corners2, False)




	    cv2.namedWindow(WINNAME)
	    cv2.imshow(WINNAME, img1)
	    key = cv2.waitKey(0)
	    cv2.imshow(WINNAME, img2)
	    key = cv2.waitKey(0)




def cluster_points (points, cluster_dist=7):
	"""
		Function: cluster_points
		------------------------
		given a list of points and the distance between them for a cluster,
		this will return a list of points with clusters compressed 
		to their centroid 
	"""
	#=====[ Step 1: init old/new points	]=====
	old_points = numpy.array (points)
	new_points = []

	#=====[ ITERATE OVER OLD_POINTS	]=====
	while len(old_points) > 1:
		p1 = old_points [0]
		distances = numpy.array([euclidean_distance (p1, p2) for p2 in old_points])
		idx = (distances < cluster_dist)
		points_cluster = old_points[idx]
		centroid = get_centroid (points_cluster)
		new_points.append (centroid)
		old_points = old_points[numpy.invert(idx)]

	return new_points

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
