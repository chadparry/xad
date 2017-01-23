#!/usr/bin/env python

import collections
import cv2
import numpy
import skimage.measure

cap = cv2.VideoCapture('idaho.webm')
#cap = cv2.VideoCapture('/usr/local/src/CVChess/data/videos/1.mov')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12,12))

IMG_WEIGHT = 0.5
MOTION_WEIGHT = 0.5
LAYER_TRAIL = 3
HISTORY_LEN = 12

layers = collections.deque()
history = collections.deque()

#for skip in range(100):
#    cap.read()

ret, first = cap.read()
#stable = numpy.zeros_like(first)
stable = first

while(1):
    ret, frame = cap.read()
    if frame is None:
        break
    changed = cv2.bitwise_xor(frame, stable)

    if len(history) >= HISTORY_LEN:
        history.pop()

    movements = numpy.zeros(stable.shape, dtype=numpy.uint8)
    for c in history:
        moved = cv2.bitwise_xor(c, changed)
        #movedmag = cv2.cvtColor(moved, cv2.COLOR_BGR2GRAY)
        movements = cv2.add(movements, moved)
        #movedmag = cv2.add(cv2.add(moved[...,0], moved[...,1]), moved[...,2])
        #(ret, movedmask) = cv2.threshold(movedmag, 16, 255, cv2.THRESH_BINARY_INV)
        #movedmaskbgr = cv2.cvtColor(movedmask, cv2.COLOR_GRAY2BGR)
        #placements = cv2.bitwise_and(movedmaskbgr, placements)
    movementsgray = cv2.cvtColor(movements.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
    (ret, grayp) = cv2.threshold(movementsgray, 64, 255, cv2.THRESH_BINARY_INV)
    placements = cv2.cvtColor(grayp, cv2.COLOR_GRAY2BGR)

    history.appendleft(changed)

    #grayp = cv2.cvtColor(placements, cv2.COLOR_BGR2GRAY)
    (ret, mask) = cv2.threshold(grayp, 1, 255, cv2.THRESH_BINARY)
    #stable[mask] = frame
    #opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    bgrmask = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    newstable = cv2.bitwise_and(frame, bgrmask)
    invmask = cv2.bitwise_not(bgrmask)
    hole = cv2.bitwise_and(stable, invmask)
    stable = cv2.bitwise_or(hole, newstable)

    history[0] = cv2.bitwise_and(history[0], invmask)

    cv2.imshow('frame', stable)
    #cv2.imshow('frame', closed)
    #cv2.imshow('frame', changed)

    writer.write(stable)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
