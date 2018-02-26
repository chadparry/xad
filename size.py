import chess

# Dimensions are taken from the FIDE Handbook,
# https://www.fide.com/FIDE/handbook/Standards_of_Chess_Equipment_and_tournament_venue.pdf

SQUARE_SIZE_MM = 57.15

HEIGHT_VARIATION = 1.1

HEIGHTS = {
	chess.KING: 95. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
	chess.QUEEN: 85. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
	chess.BISHOP: 70. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
	chess.KNIGHT: 60. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
	chess.ROOK: 55. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
	chess.PAWN: 50. * HEIGHT_VARIATION / SQUARE_SIZE_MM,
}
