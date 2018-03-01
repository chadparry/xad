#!/usr/bin/env python3

import collections
import cv2
import numpy

import subtractor
import detectmovement


HISTORY_LEN = 10


def main():
	cap = cv2.VideoCapture('idaho.webm')
	cap.set(cv2.CAP_PROP_POS_MSEC, 50000)

	lastmovelab = None
	history = collections.deque()
	while True:
		ret, framergb = cap.read()
		if framergb is None:
			break
		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		history.appendleft(framelab)
		if len(history) > HISTORY_LEN:
			history.pop()
		if len(history) < HISTORY_LEN:
			continue
		if lastmovelab is None:
			lastmovelab = framelab

		stablelab = subtractor.get_stable(lastmovelab, history)
		if lastmovelab is None:
			stablec = stablelab
		else:
			stablec = cv2.absdiff(stablelab, lastmovelab)
		stablecgray = subtractor.lab2mag(stablec).astype(numpy.uint8)

		cv2.imshow('frame', stablecgray)

		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
		if k == ord(' '):
			cv2.imwrite('subtractor.png', stablecgray)
			lastmovelab = stablelab

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
