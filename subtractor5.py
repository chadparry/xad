#!/usr/bin/env python

import collections
import cv2
import math
import numpy
import numpy.linalg
import scipy.spatial.distance
import skimage.measure

def lab2mag(img):
    return numpy.linalg.norm(img, axis=2)

def bin2mask(img):
    return cv2.merge((img, img, img))

cap = cv2.VideoCapture('idaho.webm')
#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

#ret, color1 = cap.read()
#gray1 = cv2.cvtColor(color1, cv2.COLOR_BGR2GRAY)
#pattern_size = (7, 7)
#(found1, corners1) = cv2.findChessboardCorners(gray1, pattern_size)
#cv2.drawChessboardCorners(gray1, pattern_size, corners1, False)
#cv2.imshow('frame', gray1)
#cv2.waitKey(0)

for skip in range(1000):
    cap.read()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

est_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
est_close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (32,32))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
stable_open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))

IMG_WEIGHT = 0.5
LAYER_TRAIL = 3
HISTORY_LEN = 10
RELAXED_HISTORY_LEN = 3

layers = collections.deque()
history = collections.deque()
placed_history = collections.deque()

ret, first = cap.read()
#stable = numpy.zeros_like(first)
stablergb = first
stablelabinit = cv2.cvtColor(stablergb, cv2.COLOR_BGR2LAB)
#stablelab = cv2.merge((stablelabinit[...,0], stablelabinit[...,1], stablelabinit[...,2] * 3))
stablelab = stablelabinit

laststablelab = stablelab
lastmovelab = stablelab
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, framergb = cap.read()
    if framergb is None:
        break

    framelabinit = cv2.cvtColor(framergb, cv2.COLOR_BGR2LAB)
    #framelab = cv2.merge((framelabinit[...,0], framelabinit[...,1], framelabinit[...,2] * 3))
    framelab = framelabinit

    changed = cv2.absdiff(framelab, stablelab)

    if len(history) >= HISTORY_LEN:
        history.pop()

    movements = numpy.zeros(stablelab.shape, dtype=numpy.float32)
    movementsr = movements
    for (cidx, c) in enumerate(history):
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

    #fgmask = fgbg.apply(stablergb)
    #compositefg = cv2.merge((z, z, fgmask)) + (stablergb / 2)

    # TODO: Instead of detecting color changes, should I find edges and then detect changes?

    #stablec = cv2.absdiff(stablelab, laststablelab)
    stablec = cv2.absdiff(stablelab, lastmovelab)
    stablecgrayf = lab2mag(stablec)
    stablecgray = stablecgrayf.astype(numpy.uint8)
    stablecbins = []
    for t in [4, 6, 8]:
        (ret, stablecbin) = cv2.threshold(stablecgray, t, 255, cv2.THRESH_BINARY)
        stablecopened = cv2.morphologyEx(stablecbin, cv2.MORPH_OPEN, stable_open_kernel)
        stablecmask = bin2mask(stablecopened)
        stablecbins.append(stablecopened)

    # FIXME: Is changedm better than changed?
    changedm = cv2.bitwise_and(changed, invmask)
    history.appendleft(changedm)
    laststablelab = stablelab

    z = numpy.zeros_like(stablecgray)
    #composite = (stablecmask / 2) + (stablergb / 2)
    stablergbhalf = stablergb / 2
    #composite = cv2.merge((stablergbhalf[...,0], stablergbhalf[...,1], stablecgray * 8))
    composite = (stablergb / 2) + cv2.merge((stablecbins[0] / 2, z, z)) + cv2.merge((z, stablecbins[1] / 2, z)) + cv2.merge((z, z, stablecbins[2] / 2))
    stablegray = cv2.cvtColor(stablergb, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(stablegray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,0)
    #im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(composite, contours, -1, 255, 2)

    #params = cv2.SimpleBlobDetector_Params()
    #detector = cv2.SimpleBlobDetector_create(params)
    #keypoints = detector.detect(stablergb)
    #blobs = cv2.drawKeypoints(stablergb, keypoints, numpy.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('frame', stablergb)
    #cv2.imshow('frame', stablecgray)
    #cv2.imshow('frame', composite)

    #writer.write(composite)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    if k == ord(' '):
        lastmovelab = stablelab

cap.release()
writer.release()
cv2.destroyAllWindows()
