#!/usr/bin/env python3

import chess
import collections
import cv2
import numpy

import pose
import subtractor
import detectmovement


HISTORY_LEN = 10


def main():
	# Projection found by findboard
	projection = pose.Projection(
		pose.CameraIntrinsics(
			cameraMatrix=numpy.float32([
				[887.09763773,   0.        , 639.5       ],
				[  0.        , 887.09763773, 359.5       ],
				[  0.        ,   0.        ,   1.        ],
			]),
			distCoeffs=numpy.float32([0., 0., 0., 0., 0.]),
		),
		pose.Pose(
			rvec=numpy.float32([
				[ 1.32300998],
				[-1.32785091],
				[ 1.14510022],
			]),
			tvec=numpy.float32([
				[ 3.58316198],
				[ 3.06215196],
				[10.00036672],
			])
		),
	)

	cap = cv2.VideoCapture('idaho.webm')
	cap.set(cv2.CAP_PROP_POS_MSEC, 50000)

	ret, firstrgb = cap.read()
	if firstrgb is None:
		return
	lastmovelab = cv2.cvtColor(firstrgb, cv2.COLOR_BGR2LAB)

	frame_size = tuple(reversed(lastmovelab.shape[:-1]))
	heatmaps = detectmovement.get_piece_heatmaps(frame_size, projection)
	depths = detectmovement.get_depths(projection)
	board = chess.Board()
	move_diffs = detectmovement.get_move_diffs(frame_size, heatmaps, depths, board)

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

		stablelab = subtractor.get_stable(lastmovelab, history)
		if lastmovelab is None:
			stablec = stablelab
		else:
			stablec = cv2.absdiff(stablelab, lastmovelab)
		stablecgray = subtractor.lab2mag(stablec)

		cv2.imshow('frame', stablecgray / 255)

		k = cv2.waitKey(1) & 0xff
		if k == 27:
			break
		if k == ord(' '):
			normalized_subtractor = detectmovement.normalize_diff(stablecgray)
			best_score = None
			best_move = None
			for (move, move_diff) in move_diffs.items():
				# The Pearson correlation coefficient measures the goodness of fit
				score = (move_diff * normalized_subtractor).mean()
				if best_score is None or score > best_score:
					best_score = score
					best_move = move

			print('move', board.san(best_move), best_score)
			board.push(best_move)
			lastmovelab = stablelab
			move_diffs = detectmovement.get_move_diffs(frame_size, heatmaps, depths, board)


	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
