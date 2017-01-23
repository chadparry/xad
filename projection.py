#!/usr/bin/env python

import collections
import cv2
import math
import numpy
import numpy.linalg
import scipy.spatial.distance
import skimage.measure
import sys

def lab2mag(img):
    return numpy.linalg.norm(img, axis=2)

def bin2mask(img):
    return cv2.merge((img, img, img))

cap = cv2.VideoCapture('idaho.webm')
#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

ret, color1 = cap.read()
gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
pattern_size = (3, 4)
(found1, corners1) = cv2.findChessboardCorners(gray1, pattern_size)
color1_corners = numpy.copy(color1)
#cv2.drawChessboardCorners(color1_corners, pattern_size, corners1, False)
#cv2.imshow('frame', color1_corners)
#cv2.waitKey(0)
c_slice = [c[0] for c in corners1]
refimgbw = cv2.imread('chessboard.png',0)
refimg = cv2.cvtColor(refimgbw, cv2.COLOR_GRAY2BGR)
pattern_size = (7, 7)
(found_ref, ref_chess) = cv2.findChessboardCorners(refimg, pattern_size)
ref_corners = [c[0] for c in ref_chess]
# These exact chessboard corners get found in this exact order for this particular image.
# It's just a hack until I work on the chessboard recognition.
ref_indices = [9, 8, 18, 10, 17, 25, 24, 16, 23, 15, 22 ]
clus_corners = corners = [c[0] for c in corners1]
src_pts = numpy.float32([ ref_corners[idx] for idx in ref_indices ]).reshape(-1,1,2)
dst_pts = numpy.float32([ pt for pt in clus_corners ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([src_pts], [dst_pts], gray1.shape[::-1], None, None)


warped = cv2.warpPerspective(refimg, M, (color1.shape[1], color1.shape[0]))
flatmask = numpy.full(refimg.shape, 255, dtype=numpy.uint8)
warpedmask = cv2.warpPerspective(flatmask, M, (color1.shape[1], color1.shape[0]))

maskidx = (warpedmask!=0)
overlay = numpy.copy(color1)
overlay[maskidx] = warped[maskidx]

cv2.imshow('frame', overlay)
key = cv2.waitKey(0)

axis = numpy.float32([[300,0,0], [0,300,0], [0,0,-300]]).reshape(-1,3)
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, M, dist)
for pt in imgpts:
	cv2.line(overlay, (300, 300), tuple(pt.ravel()), (255,0,0), 5)

cv2.imshow('frame', overlay)
key = cv2.waitKey(0)


cap.release()
cv2.destroyAllWindows()
