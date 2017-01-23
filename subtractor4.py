#!/usr/bin/env python

import collections
import cv2
import math
import numpy
import numpy.linalg
import scipy.spatial.distance
import skimage.measure

def lab2mag(img):
    # FIXME: Add the channels together rather than getting a grayscale.
    #return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
    return numpy.linalg.norm(img, axis=2)

def bin2mask(img):
    return cv2.merge((img, img, img))

cap = cv2.VideoCapture('idaho.webm')
#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

for skip in range(1000):
    cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

est_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
est_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32,32))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
stable_erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
stable_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))

IMG_WEIGHT = 0.5
MOTION_WEIGHT = 0.5
LAYER_TRAIL = 3
HISTORY_LEN = 10
DISPLAY_HISTORY_LEN = 24
RELAXED_HISTORY_LEN = 3

layers = collections.deque()
history = collections.deque()
placed_history = collections.deque()

ret, first = cap.read()
#stable = numpy.zeros_like(first)
stablergb = first
stablelabinit = cv2.cvtColor(stablergb, cv2.COLOR_BGR2LAB)
stablelab = cv2.merge((stablelabinit[...,0], stablelabinit[...,1], stablelabinit[...,2] * 3))

laststablelab = stablelab

while(1):
    ret, framergb = cap.read()
    if framergb is None:
        break
    framelabinit = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)
    framelab = cv2.merge((framelabinit[...,0], framelabinit[...,1], framelabinit[...,2] * 3))

    #changed = cv2.bitwise_xor(framelab, stablelab)
    changed = cv2.absdiff(framelab, stablelab)

    if len(history) >= HISTORY_LEN:
        history.pop()

    movements = numpy.zeros(stablelab.shape, dtype=numpy.float32)
    movementsr = movements
    for (cidx, c) in enumerate(history):
        #moved = cv2.bitwise_xor(c, changed)
        moved = cv2.absdiff(c, changed)
        movements = cv2.add(movements, moved.astype(numpy.float32))
        if cidx <= RELAXED_HISTORY_LEN - 1:
            movementsr = movements
    movementsmag = lab2mag(movements)
    movementsmagr = lab2mag(movementsr)

    threshold = 16 * math.sqrt(len(history)) if history else 1
    (ret, estbinf) = cv2.threshold(movementsmag, threshold, 255, cv2.THRESH_BINARY_INV)
    estbin = estbinf.astype(numpy.uint8)
    opened = cv2.morphologyEx(estbin, cv2.MORPH_OPEN, est_open_kernel)
    estmask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, est_close_kernel)

    thresholdr = 8 * math.sqrt(min(len(history), RELAXED_HISTORY_LEN - 1)) if history else 1
    (ret, relaxedgrayp) = cv2.threshold(movementsmagr, thresholdr, 255, cv2.THRESH_BINARY_INV)
    (ret, relaxedmaskf) = cv2.threshold(relaxedgrayp, 1, 255, cv2.THRESH_BINARY)
    relaxedmask = relaxedmaskf.astype(numpy.uint8)
    mask = cv2.bitwise_and(estmask, relaxedmask)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    changedmask = bin2mask(closed)
    newstablelab = cv2.bitwise_and(framelab, changedmask)
    newstablergb = cv2.bitwise_and(framergb, changedmask)
    invmask = cv2.bitwise_not(changedmask)
    holelab = cv2.bitwise_and(stablelab, invmask)
    holergb = cv2.bitwise_and(stablergb, invmask)
    stablelab = cv2.bitwise_or(holelab, newstablelab)
    stablergb = cv2.bitwise_or(holergb, newstablergb)

    #stablec = cv2.bitwise_xor(stablelab, laststablelab)
    stablec = cv2.absdiff(stablelab, laststablelab)
    stablecgrayf = lab2mag(stablec)
    stablecgray = stablecgrayf.astype(numpy.uint8)
    (ret, stablecbin) = cv2.threshold(stablecgray, 16, 255, cv2.THRESH_BINARY)
    stablecopened = cv2.morphologyEx(stablecbin, cv2.MORPH_ERODE, stable_erode_kernel)
    stablecmask = bin2mask(stablecopened)

    if len(placed_history) >= DISPLAY_HISTORY_LEN:
        placed_history.pop()
    placed_history.appendleft(stablecmask)
    placed_composite = numpy.zeros_like(stablecmask)
    for h in placed_history:
        placed_composite = cv2.bitwise_or(placed_composite, h)

    placedmask = cv2.morphologyEx(placed_composite, cv2.MORPH_DILATE, stable_dilate_kernel)
    placedlab = cv2.bitwise_and(stablelab, placedmask)
    placedrgb = cv2.bitwise_and(stablergb, placedmask)

    # FIXME: Is changedm better than changed?
    changedm = cv2.bitwise_and(changed, invmask)
    history.appendleft(changedm)
    laststablelab = stablelab



    #cv2.imshow('frame', stablergb)
    #cv2.imshow('frame', stablecmask)
    cv2.imshow('frame', placedrgb)
    #cv2.imshow('frame', changedstretch)
    #cv2.imshow('frame', relaxedmask)

    writer.write(stablergb)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
