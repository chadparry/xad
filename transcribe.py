#!/usr/bin/env python

import cv2
import numpy


WINNAME = 'TRANSCRIPTION'


def main():
	#webcam = cv2.VideoCapture(0)
	webcam = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')
	if not webcam.isOpened():
		raise RuntimeError('Failed to open camera')
	#[webcam.read() for i in range(500)]
	cv2.namedWindow(WINNAME)

	(retval, frame) = webcam.read()

	#gray = numpy.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
	#corners = cv2.cornerHarris(gray, 2, 3, 0.1)
	#cv2.imshow(WINNAME, corners)
	#key = cv2.waitKey(10000)
	#return

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	pattern_size = (7, 7)
	# FIXME: This isn't robust against occlusions
	(found, corners) = cv2.findChessboardCorners(gray, pattern_size)
	cv2.drawChessboardCorners(frame, pattern_size, corners, False)
	print('found', found)
	#print('corners', len(corners), len(corners[0]), len(corners[0][0]), corners)
	cv2.imshow(WINNAME, frame)
	key = cv2.waitKey(10000)


if __name__ == "__main__":
	main()
