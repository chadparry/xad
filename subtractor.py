#!/usr/bin/env python3

import collections
import cv2
import itertools
import math
import numpy
import numpy.linalg


EST_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
EST_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32,32))
STABLE_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))


def lab2mag(img):
	return numpy.linalg.norm(img, axis=2) / math.sqrt(3)


def bin2mask(img):
	return cv2.merge([img] * 3)


def get_stable(lastmovelab, history):
	# TODO: Use exponential backoff
	RELAXED_HISTORY_LEN = 3

	latestlab = history[0]
	movements = numpy.zeros(lastmovelab.shape, dtype=numpy.uint32)
	movementsr = movements
	for (cidx, c) in itertools.islice(enumerate(history), 1, None):
		moved = cv2.absdiff(c, latestlab)
		movements = movements + moved
		if cidx <= RELAXED_HISTORY_LEN - 1:
			movementsr = movements

	movementsmag = lab2mag(movements)
	movementsmagr = lab2mag(movementsr)

	threshold = 16 * math.sqrt(len(history)) if history else 1
	(ret, estbinf) = cv2.threshold(movementsmag, threshold, 255, cv2.THRESH_BINARY_INV)
	estbin = estbinf.astype(numpy.uint8)
	opened = cv2.morphologyEx(estbin, cv2.MORPH_OPEN, EST_OPEN_KERNEL)
	estmask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, EST_CLOSE_KERNEL)

	thresholdr = 8 * math.sqrt(min(len(history), RELAXED_HISTORY_LEN)) if history else 1
	(ret, relaxedgray) = cv2.threshold(movementsmagr, thresholdr, 255, cv2.THRESH_BINARY_INV)
	relaxedmask = relaxedgray.astype(numpy.uint8)
	mask = cv2.bitwise_and(estmask, relaxedmask)
	closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, STABLE_CLOSE_KERNEL)

	changedmask = bin2mask(closed)
	newstablelab = cv2.bitwise_and(latestlab, changedmask)
	invmask = cv2.bitwise_not(changedmask)
	holelab = cv2.bitwise_and(lastmovelab, invmask)
	stablelab = cv2.bitwise_or(holelab, newstablelab)

	return stablelab


def main():
	cap = cv2.VideoCapture('idaho.webm')
	#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

	for skip in range(1000):
		cap.read()

	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	writer = cv2.VideoWriter("subtractor.riff", fourcc, 25, (1280, 720))

	HISTORY_LEN = 10
	history = collections.deque()

	ret, first = cap.read()
	stablelab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)

	lastmovelab = stablelab
	fgbg = cv2.createBackgroundSubtractorMOG2()

	while True:
		ret, framergb = cap.read()
		if framergb is None:
			break

		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		changed = cv2.absdiff(framelab, stablelab)

		if len(history) >= HISTORY_LEN:
			history.pop()

		history.appendleft(framelab)

		if len(history) < HISTORY_LEN:
			continue

		stablelab = get_stable(lastmovelab, history)

		stablec = cv2.absdiff(stablelab, lastmovelab)
		stablecgrayf = lab2mag(stablec)
		stablecgray = stablecgrayf.astype(numpy.uint8)

		cv2.imshow('frame', stablecgray)
		#cv2.imshow('frame', composite)

		writer.write(stablecgray)
		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
		if k == ord(' '):
			cv2.imwrite('subtractor.png', stablecgray)
			lastmovelab = stablelab

	cap.release()
	writer.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
