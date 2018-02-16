#!/usr/bin/env python

from __future__ import print_function

import cv2
import functools
import itertools
import math
import numpy
import operator
import scipy.stats


WINNAME = 'Chess Transcription'


def get_piece_voxels(horizontal_resolution, vertical_resolution):
	"""Plots a piece in a 3D matrix according to the likelihood that it appears in that location"""
	shape = (vertical_resolution * 2 + 1, horizontal_resolution, horizontal_resolution)
	memoized_radial_attenuation = get_memoized_radial_attenuation(horizontal_resolution)
	reversed_idxs = itertools.product(*(xrange(dim) for dim in shape))
	idxs = (list(reversed(idx)) for idx in reversed_idxs)
	weights = (get_voxel_weight(memoized_radial_attenuation, idx, get_coords(idx, horizontal_resolution, vertical_resolution))
		for idx in idxs)
	voxel_count = reduce(operator.mul, shape, 1)
	voxels = numpy.fromiter(weights, dtype=float, count=voxel_count).reshape(shape)
	return voxels


def get_voxel_weight(memoized_radial_attenuation, idx, coords):
	getters = [memoized_radial_attenuation, height_attenuation, reflective_attenuation]
	factors = (getter(idx, *coords) for getter in getters)
	weight = reduce(operator.mul, factors, 1.)
	return weight


def get_horizontal_coord(h, horizontal_resolution):
	return (h + 1) / float(horizontal_resolution + 1)


def get_coords(idx, horizontal_resolution, vertical_resolution):
	return (
		get_horizontal_coord(idx[0], horizontal_resolution),
		get_horizontal_coord(idx[1], horizontal_resolution),
		(idx[2] - vertical_resolution) / float(vertical_resolution + 1),
	)


def get_memoized_radial_attenuation(horizontal_resolution):
	idxs = ((x, y) for y in xrange((horizontal_resolution + 1) // 2) for x in xrange(y + 1))
	memo = {(x, y):
		get_clamped_gaussian(numpy.linalg.norm([
			0.5 - get_horizontal_coord(h, horizontal_resolution)
			for h in (x, y)
		]))
		for (x, y) in idxs}
	def memoized_radial_attenuation(idx, x, y, z):
		# Take advantage of the 8-fold symmetry of the radial distance
		(mirrored_x, mirrored_y) = ((horizontal_resolution - 1 - abs(horizontal_resolution - 1 - h * 2)) // 2 for h in idx[:2])
		result = memo[(min(mirrored_x, mirrored_y), max(mirrored_x, mirrored_y))]
		return result
	return memoized_radial_attenuation



GAUSSIAN_SCALE = 3.
GAUSSION_DENOMINATOR = 1. - 2. * scipy.stats.norm.cdf(-numpy.linalg.norm((0.5, 0.5)) * GAUSSIAN_SCALE)
def get_clamped_gaussian(h):
	return scipy.stats.norm.pdf(h * GAUSSIAN_SCALE) / GAUSSION_DENOMINATOR



def height_attenuation(idx, x, y, z):
	return 1. - abs(z)


def reflective_attenuation(idx, x, y, z):
	return 0.3 if z <= 0 else 1.0
	#return 1.0


def main():
	voxels = get_piece_voxels(40, 80)
	orthographic = (voxels * 0.1).sum(axis=1)
	cv2.imshow(WINNAME, orthographic)
	key = cv2.waitKey(0)


if __name__ == "__main__":
	main()
