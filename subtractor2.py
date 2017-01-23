#!/usr/bin/env python

import collections
import cv2
import numpy
import skimage.measure

cap = cv2.VideoCapture('idaho.webm')
#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

IMG_WEIGHT = 0.5
MOTION_WEIGHT = 0.5
LAYER_TRAIL = 3
HISTORY_LEN = 12

layers = collections.deque()
history = collections.deque()

for skip in range(100):
    cap.read()

ret, first = cap.read()
#stable = numpy.zeros_like(first)
stable = first

while(1):
    ret, frame = cap.read()

    history.appendleft(frame)
    if len(history) > HISTORY_LEN:
        history.pop()

    movements = numpy.zeros(stable.shape)
    placements = None
    for f in history:
        c = cv2.bitwise_xor(f, stable)
        if placements is None:
            placements = c
        else:
            #placements = cv2.bitwise_and(c, placements)
            moved = cv2.bitwise_xor(c, placements)
            movedmag = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
            (ret, movedmask) = cv2.threshold(movedmag, 16, 255, cv2.THRESH_BINARY_INV)
            movedmaskbgr = cv2.cvtColor(movedmask, cv2.COLOR_GRAY2BGR)
            placements = cv2.bitwise_and(movedmaskbgr, placements)
        #(ret, placementbin) = cv2.threshold(placements, 1, 255, cv2.THRESH_BINARY)
        accumulated = placements.astype(numpy.float64) / HISTORY_LEN
        movements = cv2.add(accumulated, movements)

    #mask = (placements!=0)
    grayp = cv2.cvtColor(placements, cv2.COLOR_BGR2GRAY)
    (ret, mask) = cv2.threshold(grayp, 1, 255, cv2.THRESH_BINARY)
    #stable[mask] = frame
    bgrmask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    newstable = cv2.bitwise_and(frame, bgrmask)
    invmask = cv2.bitwise_not(bgrmask)
    hole = cv2.bitwise_and(stable, invmask)
    stable = cv2.bitwise_or(hole, newstable)

    intm = movements.astype(numpy.uint8)
    graym = cv2.cvtColor(intm, cv2.COLOR_BGR2GRAY)
    #graym = mask
    opening = cv2.morphologyEx(graym, cv2.MORPH_OPEN, kernel)
    gbr = numpy.zeros_like(stable)
    gbr[...,1] = opening
    gbr[...,2] = opening

    weighted = (gbr * MOTION_WEIGHT).astype(numpy.uint8)

    layers.appendleft(weighted)
    if len(layers) > LAYER_TRAIL:
        layers.pop()

    #composite = (frame * IMG_WEIGHT).astype(numpy.uint8)
    composite = (stable * IMG_WEIGHT).astype(numpy.uint8)
    for layer in layers:
        composite = cv2.bitwise_or(composite, layer)

    #cv2.imshow('frame', composite)
    cv2.imshow('frame', stable)
    #cv2.imshow('frame', bgrmask)

    writer.write(stable)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
