#!/usr/bin/env python3

import chess
import collections
import cv2
import numpy

import pose
import subtractor
import detectmovement


Particle = collections.namedtuple('Particle', ['weight', 'board', 'stablelab', 'diffs'])


EXPECTED_CORRELATION = 0.3
MAX_WEIGHT_RATIO = 0.95
HISTORY_LEN = 10


def get_board_key(board):
	return board.fen()


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
	firstlab = cv2.cvtColor(firstrgb, cv2.COLOR_BGR2LAB)

	frame_size = tuple(reversed(firstlab.shape[:-1]))
	projection_shape = tuple(reversed(frame_size))
	heatmaps = detectmovement.get_piece_heatmaps(frame_size, projection)
	depths = detectmovement.get_depths(projection)
	first_board = chess.Board()
	negative_composite_memo = {(): numpy.ones(projection_shape)}
	first_move_diffs = detectmovement.get_move_diffs(heatmaps, depths, negative_composite_memo, first_board)
	first_weight = 1.
	first_particle = Particle(first_weight, first_board, firstlab, first_move_diffs)
	particles = {get_board_key(first_board): first_particle}
	threshold_weight = first_weight * MAX_WEIGHT_RATIO

	history = collections.deque()
	while True:
		ret, framergb = cap.read()
		if framergb is None:
			break
		cv2.imshow('frame', framergb)
		key = cv2.waitKey(1)
		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		history.appendleft(framelab)
		if len(history) > HISTORY_LEN:
			history.pop()
		if len(history) < HISTORY_LEN:
			continue

		# FIXME: This needs to be get_stable_mask, because we don't know the lastmovelab!
		stablelab = subtractor.get_stable(firstlab, history)

		# FIXME: The particles are being updated inside this loop
		# Instead, create a separate collection for the next generation of particles
		for particle in list(particles.values()):

			stablec = cv2.absdiff(stablelab, particle.stablelab)
			stablecgray = subtractor.lab2mag(stablec)
			normalized_subtractor = detectmovement.normalize_diff(stablecgray)
			for (move, move_diff) in particle.diffs.items():
				# The Pearson correlation coefficient measures the goodness of fit
				score = (move_diff * normalized_subtractor).mean()
				weight = particle.weight * (score + 1 - EXPECTED_CORRELATION)
				if weight < threshold_weight:
					#print('  rejected candidate', weight, chess.Board().variation_san(nextboard.move_stack))
					continue
				nextboard = particle.board.copy()
				nextboard.push(move)
				next_particle_key = get_board_key(nextboard)
				existing_particle = particles.get(next_particle_key)
				if existing_particle is None:
					print('    getting diffs');
					next_move_diffs = detectmovement.get_move_diffs(heatmaps, depths, negative_composite_memo, nextboard)
					print('    got diffs');
					print('    negative_composite_memo', len(negative_composite_memo))
					next_particle = Particle(weight, nextboard, stablelab, next_move_diffs)
					particles[next_particle_key] = next_particle
					print('  accepted candidate', weight, chess.Board().variation_san(nextboard.move_stack))
				elif existing_particle.weight < weight:
					next_particle = Particle(weight, nextboard, stablelab, existing_particle.diffs)
					particles[next_particle_key] = next_particle
					particles[next_particle_key] = next_particle
					print('  accepted candidate', weight, chess.Board().variation_san(nextboard.move_stack))
				else:
					print('  rejected candidate', weight, chess.Board().variation_san(nextboard.move_stack))

		# Resample by removing low-weighted particles
		max_weight = max(particle.weight for particle in particles.values())
		threshold_weight = max_weight * MAX_WEIGHT_RATIO
		#print('threshold_weight', threshold_weight)
		particles = {key: particle for (key, particle) in particles.items() if particle.weight >= threshold_weight}
		print('negative_composite_memo', len(negative_composite_memo))
		print('particles', len(particles))
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			print(' ', particle.weight, chess.Board().variation_san(particle.board.move_stack))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
