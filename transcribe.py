#!/usr/bin/env python3

import chess
import chess.pgn
import collections
import cv2
import hashlib
import io
import math
import numpy
import sklearn.naive_bayes

import findboard
import pose
import subtractor
import surface
import heatmaps


Particle = collections.namedtuple('Particle', ['weight', 'board', 'stable_classes', 'diffs'])


MIN_CORRELATION = 0.1
EXPECTED_CORRELATION = 0.25
MAX_WEIGHT_RATIO = 0.75
HISTORY_LEN = 8
VOXEL_RESOLUTION = (12, 24)


WINNAME = 'Chess Transcription'


TEST_LINES = [
#	chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 c6')).end().board(),
#	chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6')).end().board(),
	chess.pgn.read_game(io.StringIO('1. e4 c5')).end().board(),
]



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
	#cv2.imshow(WINNAME, firstrgb)
	#cv2.waitKey(0)
	#while True:
	#	ret, firstrgb = cap.read()
	#	if firstrgb is None:
	#		return
	#	cv2.imshow(WINNAME, firstrgb)
	#	key = cv2.waitKey(1000 // 30) & 0xff
	#	if key == ord(' '):
	#		break

	# FIXME
	#corners = findboard.find_chessboard_corners(firstrgb)
	#projection = findboard.get_projection(corners, firstrgb.shape)
	projection = pose.Projection(cameraIntrinsics=pose.CameraIntrinsics(cameraMatrix=numpy.array([[1.60091387e+03, 0.00000000e+00, 6.39500000e+02], [0.00000000e+00, 1.60091387e+03, 3.59500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), distCoeffs=numpy.array([[0.], [0.], [0.], [0.], [0.]])), pose=pose.Pose(rvec=numpy.array([[ 1.68565304], [-1.07984874], [ 0.79226459]]), tvec=numpy.array([[ 2.72539522], [ 6.02566887], [26.47004221]])))

	#cap.set(cv2.CAP_PROP_POS_MSEC, 52000)
	#ret, firstrgb = cap.read()

	firstlab = cv2.cvtColor(firstrgb, cv2.COLOR_BGR2LAB)
	frame_size = tuple(reversed(firstlab.shape[:-1]))
	projection_shape = tuple(reversed(frame_size))

	piece_heatmaps = heatmaps.get_piece_heatmaps(frame_size, VOXEL_RESOLUTION, projection)
	depths = heatmaps.get_depths(projection)
	reference_heatmap = heatmaps.get_reference_heatmap(piece_heatmaps)
	reference_heatmap_numpy = numpy.tile(numpy.expand_dims(reference_heatmap.as_numpy(), axis=-1), (4,))
	#cv2.imshow(WINNAME, reference_heatmap.as_numpy() / reference_heatmap.delegate.max())
	#cv2.waitKey(0)
	first_board = chess.Board()

	white_pieces_board = chess.Board()
	white_pieces_board.set_piece_map({square: piece for (square, piece) in first_board.piece_map().items() if piece.color == chess.WHITE})
	black_pieces_board = chess.Board()
	black_pieces_board.set_piece_map({square: piece for (square, piece) in first_board.piece_map().items() if piece.color == chess.BLACK})
	# FIXME: Account for occlusion of pieces
	visible_white_pieces = heatmaps.get_board_heatmap(piece_heatmaps, white_pieces_board)
	visible_black_pieces = heatmaps.get_board_heatmap(piece_heatmaps, black_pieces_board)

	first_lightness = firstlab[:,:,0]
	white_average = numpy.average(first_lightness[visible_white_pieces.slice], weights=visible_white_pieces.delegate)
	black_average = numpy.average(first_lightness[visible_black_pieces.slice], weights=visible_black_pieces.delegate)

	if black_average > white_average:
		# Switch white and black
		projection = findboard.flip_sides(projection)
		piece_heatmaps = heatmaps.flip_piece_heatmaps(piece_heatmaps)
		depths = heatmaps.flip_depths(depths)
		(visible_white_pieces, visible_black_pieces) = (visible_black_pieces, visible_white_pieces)

	starting_pieces = heatmaps.get_board_heatmap(piece_heatmaps, first_board)
	#cv2.imshow(WINNAME, starting_pieces.as_numpy())
	#cv2.waitKey(0)
	light_squares = surface.get_light_square_heatmap(frame_size, projection)
	visible_light_squares = heatmaps.Heatmap.blend([light_squares, starting_pieces]).subtract(starting_pieces)
	#cv2.imshow(WINNAME, visible_light_squares.as_numpy())
	#cv2.waitKey(0)
	dark_squares = surface.get_dark_square_heatmap(frame_size, projection)
	visible_dark_squares = heatmaps.Heatmap.blend([dark_squares, starting_pieces]).subtract(starting_pieces)
	#cv2.imshow(WINNAME, visible_dark_squares.as_numpy())
	#cv2.waitKey(0)

	class_numpy = [heatmap.as_numpy() for heatmap in [visible_light_squares, visible_dark_squares, visible_white_pieces, visible_black_pieces]]
	heatmaps.show_visible_composite(numpy.stack(class_numpy, axis=2), 0)
	for heatmap in [visible_white_pieces, visible_black_pieces, visible_light_squares, visible_dark_squares]:
		cv2.imshow(WINNAME, numpy.tile(numpy.expand_dims(heatmap.as_numpy(), axis=-1), (3,)) * firstrgb / 255)
		cv2.waitKey(0)
	color_classifier = sklearn.naive_bayes.GaussianNB()
	class_heatmaps = [visible_light_squares, visible_dark_squares, visible_white_pieces, visible_black_pieces]
	class_pixels = [firstlab[class_heatmap.slice].reshape(-1, firstlab.shape[-1]) for class_heatmap in class_heatmaps]
	class_weights = [class_heatmap.delegate.ravel() for class_heatmap in class_heatmaps]
	class_labels = [numpy.full(class_weight.shape, class_idx) for (class_idx, class_weight) in enumerate(class_weights)]
	color_classifier.fit(
		numpy.vstack(class_pixels),
		numpy.hstack(class_labels),
		numpy.hstack(class_weights))

	first_classes = color_classifier.predict_proba(firstlab.reshape(-1, firstlab.shape[-1])).reshape(tuple(firstlab.shape[:2]) + (4,))
	#classified_composite = numpy.stack([
	#	first_classes[:,:,0],
	#	first_classes[:,:,1] + first_classes[:,:,2],
	#	first_classes[:,:,3] + first_classes[:,:,2],
	#], axis=2)
	#cv2.imshow(WINNAME, classified_composite)
	#cv2.waitKey(0)

	composite_memo = {(): numpy.stack([
		light_squares.as_numpy(),
		dark_squares.as_numpy(),
		heatmaps.Heatmap.blank(projection_shape).as_numpy(),
		heatmaps.Heatmap.blank(projection_shape).as_numpy(),
	], axis=-1)}
	first_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, reference_heatmap, depths, composite_memo, first_board)
	#first_move_diffs = {chess.Move(chess.E2, chess.E4): first_move_diffs[chess.Move(chess.E2, chess.E4)]}

	first_weight = 1.
	first_particle = Particle(first_weight, first_board, first_classes, first_move_diffs)
	particles = {get_board_key(first_board): first_particle}
	threshold_weight = first_weight * MAX_WEIGHT_RATIO
	max_weight = first_weight

	history = collections.deque([first_classes] * HISTORY_LEN)
	while True:
		print('memory', len(particles), sum(len(particle.diffs) for particle in particles.values()), sum(move_diff.size for particle in particles.values() for move_diff in particle.diffs.values()))
		#pos = cap.get(cv2.CAP_PROP_POS_MSEC)
		#cap.set(cv2.CAP_PROP_POS_MSEC, pos + 100)

		ret, framergb = cap.read()
		if framergb is None:
			break
		#cv2.imshow(WINNAME, framergb)
		#cv2.waitKey(1)
		framelab = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)

		# TODO: Only classify the relevant sections of the image
		frame_classes = color_classifier.predict_proba(framelab.reshape(-1, framelab.shape[-1])).reshape(tuple(framelab.shape[:-1]) + (4,))
		#heatmaps.show_visible_composite(frame_classes)

		history.appendleft(frame_classes)
		if len(history) > HISTORY_LEN:
			history.pop()

		stable_mask = subtractor.get_stable_mask(history)

		display_mask = frame_classes * stable_mask
		heatmaps.show_visible_composite(display_mask)

		#continue

		# FIXME: The particles are being updated inside this loop
		# Instead, create a separate collection for the next generation of particles
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			if TEST_LINES and not any(particle.board.move_stack == line.move_stack[:len(particle.board.move_stack)] for line in TEST_LINES):
				continue
			if not particle.diffs:
				continue

			frame_diff = frame_classes - particle.stable_classes
			stable_diff = frame_diff * stable_mask

			stable_diff_masked = stable_diff * reference_heatmap_numpy

			centered_subtractor = stable_diff - numpy.expand_dims(reference_heatmap.reweight(stable_diff_masked.sum()).as_numpy(), axis=-1)

			if TEST_LINES:
				for (move, move_diff) in particle.diffs.items():
					particle.board.push(move)
					try:
						if any(particle.board.move_stack == line.move_stack[:len(particle.board.move_stack)] for line in TEST_LINES):
							pass
							#print(particle.board.move_stack)
							#heatmaps.show_visible_composite(move_diff, 250)
					finally:
						particle.board.pop()

			#masked_centered_subtractor = reference_heatmap_numpy * centered_subtractor
			#heatmaps.show_visible_composite(masked_centered_subtractor, 250)
			# FIXME
			#break

			# The Pearson correlation coefficient measures the goodness of fit
			scores = ((move, move_diff, (move_diff * centered_subtractor).sum())
				for (move, move_diff) in particle.diffs.items())
			# Each particle represents a hypothesis for which frame contains the next move
			# Particles don't need to represent multiple alternative moves associated with a single frame,
			# because the piece detection is reliable enough, given an accurate hypothesis about the timing
			best_move_item = max(scores, key=lambda item: item[2])
			(best_move, best_move_diff, best_score) = best_move_item

			subtractor_total_variance = (centered_subtractor**2).sum()
			if subtractor_total_variance:
				subtractor_denom = math.sqrt(subtractor_total_variance * centered_subtractor.size)
			else:
				subtractor_denom = centered_subtractor.size
			best_normalized_score = best_score / subtractor_denom

			#cv2.imshow(WINNAME, (best_move_diff[:,:,2] - best_move_diff[:,:,2].min()) / (best_move_diff[:,:,2].max() - best_move_diff[:,:,2].min()))
			#key = cv2.waitKey(0)

			weight = particle.weight * (best_normalized_score + 1 - EXPECTED_CORRELATION)
			if best_score < MIN_CORRELATION or weight < threshold_weight:
				#print('  rejected candidate', weight, best_move)
				continue

			stable_classes = frame_classes * stable_mask + particle.stable_classes * (1 - stable_mask)

			next_board = particle.board.copy()
			next_board.push(best_move)
			next_particle_key = get_board_key(next_board)
			existing_particle = particles.get(next_particle_key)

			if existing_particle is None:
				next_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, reference_heatmap, depths, composite_memo, next_board)
			elif existing_particle.weight < weight:
				next_move_diffs = existing_particle.diffs
			else:
				#print('  rejected candidate', weight, chess.Board().variation_san(next_board.move_stack))
				continue

			next_particle = Particle(weight, next_board, stable_classes, next_move_diffs)
			particles[next_particle_key] = next_particle
			if weight > max_weight:
				composite = numpy.zeros(framergb.shape, dtype=numpy.float32)
				composite[:,:,0] += best_move_diff[:,:,0] + best_move_diff[:,:,1]
				composite[:,:,1] += centered_subtractor[:,:,2] + centered_subtractor[:,:,3]
				composite[:,:,2] += best_move_diff[:,:,2] + best_move_diff[:,:,3]
				for channel in range(3):
					composite[:,:,channel] -= composite[:,:,channel].min()
					composite[:,:,channel] /= composite[:,:,channel].max()
				cv2.putText(
					composite,
					'SCORE: {:.3f}'.format(best_normalized_score),
					(10, 40),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(1, 1, 1),
				)
				cv2.putText(
					composite,
					chess.Board().variation_san(next_board.move_stack),
					(10, 80),
					cv2.FONT_HERSHEY_SIMPLEX,
					1,
					(1, 1, 1),
				)
				cv2.imshow(WINNAME, composite)
				cv2.waitKey(500)

		# Resample by removing low-weighted particles
		max_weight = max(particle.weight for particle in particles.values())
		threshold_weight = max_weight * MAX_WEIGHT_RATIO
		#print('threshold_weight', threshold_weight)
		particles = {key: particle for (key, particle) in particles.items() if particle.weight >= threshold_weight}
		#print('composite_memo', len(composite_memo))
		print('particles:', len(particles), ', diffs:', sum(len(particle.diffs) for particle in particles.values()),
			#', unique diffs:', len(set(hashlib.sha224(numpy.ascontiguousarray(diff.as_numpy())).hexdigest() for particle in particles.values() for diff in particle.diffs.values())),
		)
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			print('', particle.weight, chess.Board().variation_san(particle.board.move_stack))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
