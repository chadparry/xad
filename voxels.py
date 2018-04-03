#!/usr/bin/env python3

import functools
import itertools
import math
import numpy
import operator
import scipy.stats

import size


# Probability that the center of the square is covered by the piece
CENTER_BASE_FACTOR = 1.
# Lower numbers for spread out heatmaps; higher numbers for centered heatmaps
GAUSSIAN_SCALE = 2.


def get_piece_voxels(horizontal_resolution, vertical_resolution):
	"""Plots a piece in a 3D matrix according to the likelihood that it appears in that location"""
	shape = (vertical_resolution, horizontal_resolution, horizontal_resolution)
	memoized_radial_attenuation = get_memoized_radial_attenuation(horizontal_resolution)
	reversed_idxs = itertools.product(*(range(dim) for dim in shape))
	# Indices are reversed from (z, y, x) like in the shape to (x, y, z)
	idxs = (list(reversed(idx)) for idx in reversed_idxs)
	weights = (get_voxel_weight(memoized_radial_attenuation, idx, get_coords(idx, horizontal_resolution, vertical_resolution))
		for idx in idxs)
	voxel_count = numpy.prod(shape)
	voxels = numpy.fromiter(weights, dtype=numpy.float32, count=voxel_count).reshape(shape)
	return voxels


def get_voxel_weight(memoized_radial_attenuation, idx, coords):
	getters = [memoized_radial_attenuation, taper_attenuation, height_attenuation]
	factors = (getter(idx, *coords) for getter in getters)
	weight = functools.reduce(operator.mul, factors, 1.)
	return weight


def get_horizontal_coord(h, horizontal_resolution):
	return (h + 1) / (horizontal_resolution + 1)


def get_vertical_coord(v, vertical_resolution):
	return v / vertical_resolution


def get_coords(idx, horizontal_resolution, vertical_resolution):
	return (
		get_horizontal_coord(idx[0], horizontal_resolution),
		get_horizontal_coord(idx[1], horizontal_resolution),
		get_vertical_coord(idx[2], vertical_resolution),
	)


def get_memoized_radial_attenuation(horizontal_resolution):
	idxs = ((x, y) for y in range((horizontal_resolution + 1) // 2) for x in range(y + 1))
	DENSITY_FACTOR = 1 / (1 - (1 - 1 / math.sqrt(2 * math.pi)) ** (1 / CENTER_BASE_FACTOR))
	def get_clamped_gaussian(h):
		return scipy.stats.norm.pdf(h * GAUSSIAN_SCALE) * DENSITY_FACTOR
	memo = {(x, y):
		get_clamped_gaussian(numpy.linalg.norm([
			0.5 - get_horizontal_coord(h, horizontal_resolution)
			for h in (x, y)
		]))
		for (x, y) in idxs}
	def memoized_radial_attenuation(idx, x, y, z):
		# Take advantage of the 8-fold symmetry of the radial distance
		(mirrored_x, mirrored_y) = (
			(horizontal_resolution - 1 - abs(horizontal_resolution - 1 - h * 2)) // 2
			for h in idx[:2])
		result = memo[(min(mirrored_x, mirrored_y), max(mirrored_x, mirrored_y))]
		return result
	return memoized_radial_attenuation


def taper_attenuation(idx, x, y, z):
	return 1 - z * math.sqrt((0.5 - x)**2 + (0.5 - y)**2)


def height_attenuation(idx, x, y, z):
	MIN_RELATIVE_HEIGHT = 1 / size.HEIGHT_VARIATION**2
	return 1 - max(z - MIN_RELATIVE_HEIGHT, 0) / (1 - MIN_RELATIVE_HEIGHT)
