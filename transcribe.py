#!/usr/bin/env python3

import chess
import collections
import cv2
import hashlib
import math
import numpy

import findboard
import pose
import subtractor
import surface
import heatmaps


Particle = collections.namedtuple('Particle', ['weight', 'board', 'stablelab', 'diffs'])


MIN_CORRELATION = 0.35
EXPECTED_CORRELATION = 0.5
MAX_WEIGHT_RATIO = 0.75
HISTORY_LEN = 8


WINNAME = 'Chess Transcription'


def get_board_key(board):
	return board.fen()


def main():
	#cap = cv2.VideoCapture('idaho.webm')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 52000)
	# Game from https://www.youtube.com/watch?v=jOU3tmXgB8A
	#cap = cv2.VideoCapture('acerook.mp4')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 7000)
	# Game from https://www.youtube.com/watch?v=aHZtDuUMK50
	#cap = cv2.VideoCapture('armin.mp4')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 13000)
	# Game from https://www.youtube.com/watch?v=_N5sEyVc38o
	#cap = cv2.VideoCapture('Zaven Adriasian - Luke McShane, Italian game, Blitz chess.mp4')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 105000)
	# Game from https://www.youtube.com/watch?v=fot9b08TuWc
	cap = cv2.VideoCapture('Carlsen-Karjakin, World Blitz Championship 2012.mp4')
	cap.set(cv2.CAP_PROP_POS_MSEC, 11000)

	ret, firstrgb = cap.read()
	cv2.imshow(WINNAME, firstrgb)
	cv2.waitKey(1)
	#while True:
	#	ret, firstrgb = cap.read()
	#	if firstrgb is None:
	#		return
	#	cv2.imshow(WINNAME, firstrgb)
	#	key = cv2.waitKey(1000 // 30) & 0xff
	#	if key == ord(' '):
	#		break

	corners = findboard.find_chessboard_corners(firstrgb)
	projection = findboard.get_projection(corners, firstrgb.shape)

	firstlab = cv2.cvtColor(firstrgb, cv2.COLOR_BGR2LAB)
	frame_size = tuple(reversed(firstlab.shape[:-1]))

	light_squares = surface.get_light_square_heatmap(frame_size, projection)
	cv2.imshow(WINNAME, light_squares.as_dense().delegate)
	cv2.waitKey(500)
	dark_squares = surface.get_dark_square_heatmap(frame_size, projection)
	cv2.imshow(WINNAME, dark_squares.as_dense().delegate)
	cv2.waitKey(500)

	#cap.set(cv2.CAP_PROP_POS_MSEC, 52000)
	#ret, firstrgb = cap.read()

	# FIXME: Switch white and black
	projection = findboard.flip_sides(projection)

	projection_shape = tuple(reversed(frame_size))
	piece_heatmaps = heatmaps.get_piece_heatmaps(frame_size, projection)
	occlusions = heatmaps.get_occlusions(piece_heatmaps, projection)
	reference_heatmap = heatmaps.get_reference_heatmap(piece_heatmaps)
	cv2.imshow(WINNAME, reference_heatmap.as_dense().delegate * 100000)
	cv2.waitKey(1000)
	first_board = chess.Board()
	negative_composite_memo = {(): heatmaps.Heatmap.blank(projection_shape)}
	first_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, reference_heatmap, occlusions, negative_composite_memo, first_board)

	first_weight = 1.
	first_particle = Particle(first_weight, first_board, firstlab, first_move_diffs)
	particles = {get_board_key(first_board): first_particle}
	threshold_weight = first_weight * MAX_WEIGHT_RATIO
	max_weight = first_weight

	history = collections.deque([firstlab] * HISTORY_LEN)
	while True:
		#pos = cap.get(cv2.CAP_PROP_POS_MSEC)
		#cap.set(cv2.CAP_PROP_POS_MSEC, pos + 100)

		ret, framergb = cap.read()
		if framergb is None:
			break
		cv2.imshow(WINNAME, framergb)
		cv2.waitKey(1)
		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		history.appendleft(framelab)
		if len(history) > HISTORY_LEN:
			history.pop()

		stable_mask = subtractor.get_stable_mask(history)

		#display_mask = cv2.bitwise_and(framergb, stable_mask)
		#cv2.imshow(WINNAME, display_mask)
		#cv2.waitKey(1)

		# FIXME: The particles are being updated inside this loop
		# Instead, create a separate collection for the next generation of particles
		for particle in list(particles.values()):
			if not particle.diffs:
				continue

			frame_diff = cv2.absdiff(framelab, particle.stablelab)
			stable_diff = cv2.bitwise_and(frame_diff, stable_mask)

			#cv2.imshow(WINNAME, stable_diff)
			#key = cv2.waitKey(1) & 0xff
			#advance_move = key == ord(' ')
			#print('*', key, '*', advance_move)

			stable_diff_gray = subtractor.lab2mag(stable_diff)
			stable_diff_heatmap = heatmaps.Heatmap(stable_diff_gray)
			stable_diff_masked = heatmaps.Heatmap.product_zeros([reference_heatmap, stable_diff_heatmap])

			centered_subtractor = stable_diff_masked.subtract(reference_heatmap.reweight(stable_diff_masked.sum()))

			# The Pearson correlation coefficient measures the goodness of fit
			scores = ((move, move_diff, move_diff.correlation(centered_subtractor))
				for (move, move_diff) in particle.diffs.items())
			# Each particle represents a hypothesis for which frame contains the next move
			# Particles don't need to represent multiple alternative moves associated with a single frame,
			# because the piece detection is reliable enough, given an accurate hypothesis about the timing
			best_move_item = max(scores, key=lambda item: item[2])
			(best_move, best_move_diff, best_score) = best_move_item

			subtractor_total_variance = centered_subtractor.total_variance()
			if subtractor_total_variance:
				subtractor_denom = math.sqrt(subtractor_total_variance * centered_subtractor.size())
			else:
				subtractor_denom = centered_subtractor.size()
			best_normalized_score = best_score / subtractor_denom

			weight = particle.weight * (best_normalized_score + 1 - EXPECTED_CORRELATION)
			if best_score < MIN_CORRELATION or weight < threshold_weight:
				#print('  rejected candidate', weight, chess.Board().variation_san(next_board.move_stack))
				continue

			newstablelab = cv2.bitwise_and(framelab, stable_mask)
			invmask = cv2.bitwise_not(stable_mask)
			holelab = cv2.bitwise_and(particle.stablelab, invmask)
			stablelab = cv2.bitwise_or(holelab, newstablelab)

			next_board = particle.board.copy()
			next_board.push(best_move)
			next_particle_key = get_board_key(next_board)
			existing_particle = particles.get(next_particle_key)

			if existing_particle is None:
				next_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, reference_heatmap, occlusions, negative_composite_memo, next_board)
			elif existing_particle.weight < weight:
				next_move_diffs = existing_particle.diffs
			else:
				#print('  rejected candidate', weight, chess.Board().variation_san(next_board.move_stack))
				continue

			next_particle = Particle(weight, next_board, stablelab, next_move_diffs)
			particles[next_particle_key] = next_particle
			#if advance_move:
			#	particles = dict({next_particle_key: next_particle})
			#	print('    accepted candidate', best_normalized_score, chess.Board().variation_san(next_board.move_stack))

			#if advance_move:
			if weight > max_weight:
				composite = numpy.zeros(framergb.shape, dtype=numpy.float32)
				composite[:,:,2][best_move_diff.slice] += best_move_diff.delegate
				composite[:,:,1][centered_subtractor.slice] += centered_subtractor.delegate * (centered_subtractor.size() / subtractor_denom)
				cv2.putText(
					composite,
					'SCORE: {:.3f}'.format(best_normalized_score),
					(10, 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(255, 255, 255),
				)
				cv2.putText(
					composite,
					chess.Board().variation_san(next_board.move_stack),
					(10, 80),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(255, 255, 255),
				)
				cv2.imshow(WINNAME, composite / 50)
				cv2.waitKey(500)
			#advance_move = False

		# Resample by removing low-weighted particles
		max_weight = max(particle.weight for particle in particles.values())
		threshold_weight = max_weight * MAX_WEIGHT_RATIO
		#print('threshold_weight', threshold_weight)
		particles = {key: particle for (key, particle) in particles.items() if particle.weight >= threshold_weight}
		#print('negative_composite_memo', len(negative_composite_memo))
		print('particles:', len(particles), ', diffs:', sum(len(particle.diffs) for particle in particles.values()),
			#', unique diffs:', len(set(hashlib.sha224(numpy.ascontiguousarray(diff.as_numpy())).hexdigest() for particle in particles.values() for diff in particle.diffs.values())),
		)
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			print('', particle.weight, chess.Board().variation_san(particle.board.move_stack))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
