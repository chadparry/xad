#!/usr/bin/env python3

import chess
import collections
import cv2
import hashlib
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


def get_board_key(board):
	return board.fen()


def main():
	#cap = cv2.VideoCapture('idaho.webm')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 52000)
	#cap = cv2.VideoCapture('dining.webm')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 40)
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
	#cap = cv2.VideoCapture('Carlsen-Karjakin, World Blitz Championship 2012.mp4')
	#cap.set(cv2.CAP_PROP_POS_MSEC, 11000)
	cap = cv2.VideoCapture('GM Hikaru LAUGHS At New Hustler\'s IM Warning! GM Hikaru vs Pete [6hFCfRq1tVc].webm')
	cap.set(cv2.CAP_PROP_POS_MSEC, 7000)

	ret, orientbgr = cap.read()
	#cv2.imshow(WINNAME, orientbgr)
	#cv2.waitKey(1)
	#while True:
	#	ret, orientbgr = cap.read()
	#	if orientbgr is None:
	#		return
	#	cv2.imshow(WINNAME, orientbgr)
	#	key = cv2.waitKey(1000 // 30) & 0xff
	#	if key == ord(' '):
	#		break

	# FIXME
	#corners = findboard.find_chessboard_corners(orientbgr)
	#projection = findboard.get_projection(corners, orientbgr.shape)
	projection = pose.Projection(cameraIntrinsics=pose.CameraIntrinsics(cameraMatrix=numpy.array([[1.86451431e+03, 0.00000000e+00, 9.59500000e+02], [0.00000000e+00, 1.86451431e+03, 5.39500000e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]), distCoeffs=numpy.array([[0.],        [0.],        [0.],        [0.],        [0.]])), pose=pose.Pose(rvec=numpy.array([[ 1.46354967],        [ 1.49479278],        [-1.11067136]]), tvec=numpy.array([[-3.49307017],        [ 4.2543103 ],        [31.74298106]])))

	cap.set(cv2.CAP_PROP_POS_MSEC, 24000)
	ret, orientbgr = cap.read()

	frame_size = tuple(reversed(orientbgr.shape[:-1]))
	projection_shape = tuple(reversed(frame_size))

	light_squares = surface.get_light_square_heatmap(frame_size, projection)
	dark_squares = surface.get_dark_square_heatmap(frame_size, projection)

	piece_heatmaps = heatmaps.get_piece_heatmaps(frame_size, VOXEL_RESOLUTION, projection)
	depths = heatmaps.get_depths(projection)
	#reference_heatmap = heatmaps.get_reference_heatmap(piece_heatmaps)
	#reference_heatmap_numpy = numpy.tile(numpy.expand_dims(reference_heatmap.as_numpy(), axis=-1), (4,))
	interest_area = heatmaps.Heatmap.superimpose(list(piece_heatmaps.values()) + [light_squares, dark_squares])
	first_board = chess.Board()

	firstbgr = heatmaps.Heatmap(orientbgr).clip(interest_area.slice)
	firstlab = heatmaps.Heatmap(cv2.cvtColor(firstbgr.delegate, cv2.COLOR_BGR2LAB), firstbgr.slice, firstbgr.shape)

	masked_firstlab = heatmaps.Heatmap(numpy.where(interest_area.delegate[...,numpy.newaxis], firstlab.delegate, 0), firstlab.slice, firstlab.shape)
	# history = collections.deque([masked_firstlab.as_numpy()] * HISTORY_LEN)
	history = collections.deque([masked_firstlab.delegate])
	stable_frame = masked_firstlab.delegate

	white_pieces_board = chess.Board()
	white_pieces_board.set_piece_map({square: piece for (square, piece) in first_board.piece_map().items() if piece.color == chess.WHITE})
	black_pieces_board = chess.Board()
	black_pieces_board.set_piece_map({square: piece for (square, piece) in first_board.piece_map().items() if piece.color == chess.BLACK})
	# FIXME: Account for occlusion of pieces
	visible_white_pieces = heatmaps.get_board_heatmap(piece_heatmaps, white_pieces_board)
	visible_black_pieces = heatmaps.get_board_heatmap(piece_heatmaps, black_pieces_board)

	first_lightness = masked_firstlab.as_numpy()[...,0]
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
	#cv2.waitKey(500)

	visible_light_squares = heatmaps.Heatmap.blend([light_squares, starting_pieces]).subtract(starting_pieces)
	#cv2.imshow(WINNAME, visible_light_squares.as_numpy())
	#cv2.imshow(WINNAME, visible_light_squares.as_dense().delegate)
	#cv2.waitKey(0)
	visible_dark_squares = heatmaps.Heatmap.blend([dark_squares, starting_pieces]).subtract(starting_pieces)
	#cv2.imshow(WINNAME, visible_dark_squares.as_numpy())
	#cv2.imshow(WINNAME, visible_dark_squares.as_dense().delegate)
	#cv2.waitKey(0)

	color_classifier = sklearn.naive_bayes.GaussianNB()
	class_heatmaps = [visible_light_squares, visible_dark_squares, visible_white_pieces, visible_black_pieces]
	class_pixels = [masked_firstlab.clip(class_heatmap.slice).delegate.reshape(-1, masked_firstlab.shape[-1]) for class_heatmap in class_heatmaps]
	class_weights = [class_heatmap.delegate.ravel() for class_heatmap in class_heatmaps]
	class_labels = [numpy.full(class_weight.shape, class_idx) for (class_idx, class_weight) in enumerate(class_weights)]
	color_classifier.fit(
		numpy.vstack(class_pixels),
		numpy.hstack(class_labels),
		numpy.hstack(class_weights))

	first_classes = color_classifier.predict_proba(masked_firstlab.delegate.reshape(-1, masked_firstlab.delegate.shape[-1])).reshape(tuple(masked_firstlab.delegate.shape[:2]) + (4,))
	#classified_composite = numpy.stack([
	#	first_classes[...,0],
	#	first_classes[...,1] + first_classes[...,2],
	#	first_classes[...,3] + first_classes[...,2],
	#], axis=2)
	#cv2.imshow(WINNAME, classified_composite)
	#cv2.waitKey(1)

	composite_memo = {(): heatmaps.Heatmap(numpy.stack([
		light_squares.as_numpy(),
		dark_squares.as_numpy(),
		heatmaps.Heatmap.blank(projection_shape).as_numpy(),
		heatmaps.Heatmap.blank(projection_shape).as_numpy(),
	], axis=-1)).as_sparse()}
	first_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, depths, composite_memo, first_board)
	#first_move_diffs = {chess.Move(chess.E2, chess.E4): first_move_diffs[chess.Move(chess.E2, chess.E4)]}

	first_weight = 1.
	first_particle = Particle(first_weight, first_board, first_classes, first_move_diffs)
	particles = {get_board_key(first_board): first_particle}
	threshold_weight = first_weight * MAX_WEIGHT_RATIO
	max_weight = first_weight

	# FIXME: Keep track of the best weight for each ply. Only keep the one best game history, where each particle is a prefix of that game history.
	best_weights = [first_weight]
	while True:

		# Reduce memory usage by clearing the memo frequently
		composite_memo = {(): heatmaps.Heatmap(numpy.stack([
			light_squares.as_numpy(),
			dark_squares.as_numpy(),
			heatmaps.Heatmap.blank(projection_shape).as_numpy(),
			heatmaps.Heatmap.blank(projection_shape).as_numpy(),
		], axis=-1)).as_sparse()}

		#pos = cap.get(cv2.CAP_PROP_POS_MSEC)
		#cap.set(cv2.CAP_PROP_POS_MSEC, pos + 100)

		ret, readbgr = cap.read()
		if readbgr is None:
			break
		framebgr = heatmaps.Heatmap(readbgr).clip(interest_area.slice)
		cv2.imshow(WINNAME, readbgr)
		#cv2.imshow(WINNAME, framebgr.as_numpy())
		framelab = heatmaps.Heatmap(cv2.cvtColor(framebgr.delegate, cv2.COLOR_BGR2LAB), framebgr.slice, framebgr.shape)
		#cv2.imshow(WINNAME, framelab.as_numpy())
		#cv2.waitKey(0)
		masked_framelab = heatmaps.Heatmap(numpy.where(interest_area.delegate[...,numpy.newaxis], framelab.delegate, 0), framelab.slice, framelab.shape)
		#cv2.imshow(WINNAME, masked_framelab.as_numpy())
		#cv2.waitKey(1)

		history.appendleft(masked_framelab.delegate)
		if len(history) > HISTORY_LEN:
			history.pop()

		stable_mask = subtractor.get_stable_mask(history)
		stable_frame = numpy.where(stable_mask, masked_framelab.delegate, stable_frame)
		#cv2.imshow(WINNAME, stable_frame)
		#cv2.waitKey(1)

		stable_classes = color_classifier.predict_proba(stable_frame.reshape(-1, stable_frame.shape[-1])).reshape(tuple(stable_frame.shape[:-1]) + (4,))
		classified_composite = numpy.stack([
			stable_classes[...,0],
			stable_classes[...,1] + stable_classes[...,2],
			stable_classes[...,3] + stable_classes[...,2],
		], axis=2)
		masked_composite = numpy.where(stable_mask, classified_composite, 0)
		cv2.imshow(WINNAME, masked_composite)
		cv2.waitKey(1)

		# FIXME: The particles are being updated inside this loop
		# Instead, create a separate collection for the next generation of particles
		# FIXME: Run particles and piece candidates in parallel threads
		for particle in list(particles.values()):
			if not particle.diffs:
				continue

			stable_diff = stable_classes - particle.stable_classes
			#stable_composite = numpy.stack([
			#	stable_diff[...,0],
			#	stable_diff[...,1] + stable_diff[...,2],
			#	stable_diff[...,3] + stable_diff[...,2],
			#], axis=2)
			#cv2.imshow(WINNAME, stable_composite)
			#cv2.waitKey(1)

			#stable_diff_gray = subtractor.lab2mag(stable_diff)
			#stable_diff_heatmap = heatmaps.Heatmap(stable_diff_gray)
			#stable_diff_masked = heatmaps.Heatmap.product_zeros([reference_heatmap, stable_diff_heatmap])
			#stable_diff_masked = stable_diff * reference_heatmap_numpy
			#centered_subtractor = heatmaps.Heatmap(stable_diff_masked - numpy.expand_dims(reference_heatmap.reweight(stable_diff_masked.sum()).as_numpy(), axis=-1))
			centered_subtractor = heatmaps.Heatmap(stable_diff, masked_framelab.slice, masked_framelab.shape[:-1] + (4,)).normalize_sum()
			#cv2.imshow(WINNAME, (centered_subtractor.delegate[...,2] - centered_subtractor.delegate[...,2].min()) / (centered_subtractor.delegate[...,2].max() - centered_subtractor.delegate[...,2].min()))
			#key = cv2.waitKey(100)

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

			#cv2.imshow(WINNAME, (best_move_diff[...,2] - best_move_diff[...,2].min()) / (best_move_diff[...,2].max() - best_move_diff[...,2].min()))
			#key = cv2.waitKey(0)

			weight = particle.weight * (best_normalized_score + 1 - EXPECTED_CORRELATION)
			if best_score < MIN_CORRELATION or weight < threshold_weight:
				#print('  rejected candidate', weight, best_move)
				continue
			# Keep only the best particle for each ply
			if len(best_weights) > particle.board.ply() and best_weights[particle.board.ply()] >= weight:
				continue
			best_weights = best_weights[:particle.board.ply() + 1] + [weight]

			next_board = particle.board.copy()
			next_board.push(best_move)
			next_particle_key = get_board_key(next_board)
			existing_particle = particles.get(next_particle_key)

			if existing_particle is None:
				next_move_diffs = heatmaps.get_move_diffs(piece_heatmaps, depths, composite_memo, next_board)
			elif existing_particle.weight < weight:
				next_move_diffs = existing_particle.diffs
			else:
				#print('  rejected candidate', weight, chess.Board().variation_san(next_board.move_stack))
				continue

			next_particle = Particle(weight, next_board, stable_classes, next_move_diffs)
			particles[next_particle_key] = next_particle
			#if advance_move:
			#	particles = dict({next_particle_key: next_particle})
			#	print('    accepted candidate', best_normalized_score, chess.Board().variation_san(next_board.move_stack))

			#if advance_move:
			if weight > max_weight:
				print(next_board)
				composite = numpy.zeros(centered_subtractor.delegate.shape, dtype=numpy.float32)
				expanded_best_move_diff = best_move_diff.expand(centered_subtractor.slice)
				composite[...,0] += expanded_best_move_diff.delegate[...,0] + expanded_best_move_diff.delegate[...,1]
				composite[...,1] += centered_subtractor.delegate[...,2] + centered_subtractor.delegate[...,3]
				composite[...,2] += expanded_best_move_diff.delegate[...,2] + expanded_best_move_diff.delegate[...,3]
				for channel in range(3):
					composite[...,channel] -= composite[...,channel].min()
					composite[...,channel] /= composite[...,channel].max()
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
				cv2.waitKey(1000)

			#advance_move = False

		# Resample by removing low-weighted particles
		max_weight = max(particle.weight for particle in particles.values())
		threshold_weight = max_weight * MAX_WEIGHT_RATIO
		#print('threshold_weight', threshold_weight)
		particles = {key: particle for (key, particle) in particles.items() if particle.weight >= threshold_weight}
		#print('composite_memo', len(composite_memo))
		# Also keep only the best particle for each game length
		particles = {key: particle for (key, particle) in particles.items() if len(best_weights) > particle.board.ply() and particle.weight >= best_weights[particle.board.ply()]}
		print('particles:', len(particles), ', diffs:', sum(len(particle.diffs) for particle in particles.values()),
			#', unique diffs:', len(set(hashlib.sha224(numpy.ascontiguousarray(diff.as_numpy())).hexdigest() for particle in particles.values() for diff in particle.diffs.values())),
		)
		for particle in sorted(particles.values(), key=lambda particle: particle.weight, reverse=True):
			print('', particle.weight, chess.Board().variation_san(particle.board.move_stack))

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
