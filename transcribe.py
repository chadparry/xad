#!/usr/bin/env python3

import chess
import collections
import cv2
import numpy

import pose
import subtractor
import detectmovement


Particle = collections.namedtuple('Particle', ['weight', 'board', 'stablelab', 'diffs'])


MIN_CORRELATION = 0.2
EXPECTED_CORRELATION = 0.4
MAX_WEIGHT_RATIO = 0.7
HISTORY_LEN = 10


def get_board_key(board):
	return board.fen()


def main():
	answer_board = chess.Board()
	for san in [
		'e4',
		'c5',
		'Nf3',
		'e6',
		'd4',
		'cxd4',
		'Nxd4',
		'Nf6',
		'Nc3',
		'Nc6',
		'Nxc6',
		'dxc6',
		'Qxd8+',
		'Kxd8',
		'Bg5',
		'Be7',
		'O-O-O',
		'Bd7',
	]:
		answer_board.push_san(san)
	ANSWER = answer_board.move_stack

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
		pos = cap.get(cv2.CAP_PROP_POS_MSEC)
		cap.set(cv2.CAP_PROP_POS_MSEC, pos + 100)

		ret, framergb = cap.read()
		if framergb is None:
			break
		#cv2.imshow('frame', framergb)
		#cv2.waitKey(1)
		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		history.appendleft(framelab)
		if len(history) > HISTORY_LEN:
			history.pop()
		if len(history) < HISTORY_LEN:
			continue

		stable_mask = subtractor.get_stable_mask(history)

		# FIXME: The particles are being updated inside this loop
		# Instead, create a separate collection for the next generation of particles
		for particle in list(particles.values()):

			frame_diff = cv2.absdiff(framelab, particle.stablelab)
			stablec = cv2.bitwise_and(frame_diff, stable_mask)
			stablecgray = subtractor.lab2mag(stablec)
			#cv2.imshow('frame', stablecgray / 255)
			#key = cv2.waitKey(1)
			normalized_subtractor = detectmovement.normalize_diff(stablecgray)
			best_weight = None
			best_score = None
			best_board = None
			best_move_diff = None
			for (move, move_diff) in particle.diffs.items():
				# The Pearson correlation coefficient measures the goodness of fit
				score = (move_diff * normalized_subtractor).mean()
				weight = particle.weight * (score + 1 - EXPECTED_CORRELATION)
				if score < MIN_CORRELATION or weight < threshold_weight:
					#print('  rejected candidate', weight, chess.Board().variation_san(nextboard.move_stack))
					continue
				nextboard = particle.board.copy()
				nextboard.push(move)
				next_particle_key = get_board_key(nextboard)
				existing_particle = particles.get(next_particle_key)
				if best_weight is None or weight > best_weight:
					best_weight = weight
					best_score = score
					best_board = nextboard
					best_move_diff = move_diff
				if existing_particle is None:
					# Handle after the loop
					pass
				elif existing_particle.weight < weight:
					newstablelab = cv2.bitwise_and(framelab, stable_mask)
					invmask = cv2.bitwise_not(stable_mask)
					holelab = cv2.bitwise_and(particle.stablelab, invmask)
					stablelab = cv2.bitwise_or(holelab, newstablelab)

					next_particle = Particle(weight, nextboard, stablelab, existing_particle.diffs)
					particles[next_particle_key] = next_particle
					print('    accepted candidate', score, chess.Board().variation_san(nextboard.move_stack))
					if weight >= best_weight:
						best_board = None
					if next_particle.board.move_stack == ANSWER[:len(next_particle.board.move_stack)]:
						composite = numpy.zeros((720, 1280, 3), dtype='float32')
						composite[:,:,2] += move_diff
						composite[:,:,1] += normalized_subtractor
						cv2.putText(
							composite,
							'SCORE: {:.3f} , MOVE: {}'.format(
								score,
								chess.Board().variation_san(nextboard.move_stack),
							),
							(10, 40),
							cv2.FONT_HERSHEY_SIMPLEX,
							1,
							(255, 255, 255),
						)
						cv2.imshow('frame', composite / 20)
						cv2.waitKey(1)
				else:
					print('    rejected candidate', score, chess.Board().variation_san(nextboard.move_stack))
					if weight >= best_weight:
						best_board = None
			# Each particle represents a hypothesis for which frame contains the next move
			# Particles don't need to represent multiple alternative moves associated with a single frame,
			# because the piece detection is reliable enough, given an accurate frame hypothesis
			if best_board is not None:
				print('      getting diffs');
				best_move_diffs = detectmovement.get_move_diffs(heatmaps, depths, negative_composite_memo, best_board)
				print('      got diffs');
				print('      negative_composite_memo', len(negative_composite_memo))

				newstablelab = cv2.bitwise_and(framelab, stable_mask)
				invmask = cv2.bitwise_not(stable_mask)
				holelab = cv2.bitwise_and(particle.stablelab, invmask)
				stablelab = cv2.bitwise_or(holelab, newstablelab)

				best_particle = Particle(best_weight, best_board, stablelab, best_move_diffs)
				best_particle_key = get_board_key(best_board)
				particles[best_particle_key] = best_particle
				print('    accepted candidate', best_score, chess.Board().variation_san(best_board.move_stack))
				if best_particle.board.move_stack == ANSWER[:len(best_particle.board.move_stack)]:
					composite = numpy.zeros((720, 1280, 3), dtype='float32')
					composite[:,:,2] += best_move_diff
					composite[:,:,1] += normalized_subtractor
					cv2.putText(
						composite,
						'SCORE: {:.3f} , MOVE: {}'.format(
							best_score,
							chess.Board().variation_san(best_board.move_stack),
						),
						(10, 40),
						cv2.FONT_HERSHEY_SIMPLEX,
						1,
						(255, 255, 255),
					)
					cv2.imshow('frame', composite / 20)
					cv2.waitKey(1)

		# Resample by removing low-weighted particles
		max_weight = max(particle.weight for particle in particles.values())
		threshold_weight = max_weight * MAX_WEIGHT_RATIO
		#print('threshold_weight', threshold_weight)
		particles = {key: particle for (key, particle) in particles.items() if particle.weight >= threshold_weight}
		#print('negative_composite_memo', len(negative_composite_memo))
		print('particles', len(particles))
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			print(
				'*' if particle.board.move_stack == ANSWER[:len(particle.board.move_stack)] else ' ',
				particle.weight, chess.Board().variation_san(particle.board.move_stack))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
