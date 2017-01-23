#!/usr/bin/env python

import collections
import cv2
import numpy
import skimage.measure

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
stable = first
laststable = stable

while(1):
    ret, frame = cap.read()
    if frame is None:
        break
    changed = cv2.bitwise_xor(frame, stable)

    if len(history) >= HISTORY_LEN:
        history.pop()

    movementsr = None
    movements = numpy.zeros(stable.shape, dtype=numpy.float64)
    for (cidx, c) in enumerate(history):
        moved = cv2.bitwise_xor(c, changed)
        movements = cv2.add(movements, moved.astype(numpy.float64))
        if cidx == RELAXED_HISTORY_LEN - 1:
            movementsr = numpy.copy(movements)
    if history:
        movementsscaled = movements / len(history)
    else:
        movementsscaled = movements
    if movementsr is None:
        movementsscaledr = movementsscaled
    else:
        movementsscaledr = movementsr / RELAXED_HISTORY_LEN

    movementsgray = cv2.cvtColor(movementsscaled.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
    (ret, estbin) = cv2.threshold(movementsgray, 8, 255, cv2.THRESH_BINARY_INV)
    opened = cv2.morphologyEx(estbin, cv2.MORPH_OPEN, est_open_kernel)
    estmask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, est_close_kernel)

    movementsgrayr = cv2.cvtColor(movementsscaledr.astype(numpy.uint8), cv2.COLOR_BGR2GRAY)
    (ret, relaxedgrayp) = cv2.threshold(movementsgrayr, 128, 255, cv2.THRESH_BINARY_INV)
    (ret, relaxedmask) = cv2.threshold(relaxedgrayp, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(estmask, relaxedmask)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    bgrmask = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    newstable = cv2.bitwise_and(frame, bgrmask)
    invmask = cv2.bitwise_not(bgrmask)
    hole = cv2.bitwise_and(stable, invmask)
    stable = cv2.bitwise_or(hole, newstable)

    stablec = cv2.bitwise_xor(stable, laststable)
    stablecgray = cv2.cvtColor(stablec, cv2.COLOR_BGR2GRAY)
    (ret, stablecbin) = cv2.threshold(stablecgray, 64, 255, cv2.THRESH_BINARY)
    stablecopened = cv2.morphologyEx(stablecbin, cv2.MORPH_ERODE, stable_erode_kernel)
    stablecmask = cv2.cvtColor(stablecopened, cv2.COLOR_GRAY2BGR)

    if len(placed_history) >= DISPLAY_HISTORY_LEN:
        placed_history.pop()
    placed_history.appendleft(stablecmask)
    placed_composite = numpy.zeros_like(stablecmask)
    for h in placed_history:
        placed_composite = cv2.bitwise_or(placed_composite, h)

    placedmask = cv2.morphologyEx(placed_composite, cv2.MORPH_DILATE, stable_erode_kernel)
    placed = cv2.bitwise_and(stable, placedmask)

    # FIXME: Is changedm better than changed?
    changedm = cv2.bitwise_and(changed, invmask)
    history.appendleft(changedm)
    laststable = stable

    cv2.imshow('frame', stable)
    #cv2.imshow('frame', placed)
    #cv2.imshow('frame', placed_composite)
    #cv2.imshow('frame', movementsgrayr)

    writer.write(stable)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
