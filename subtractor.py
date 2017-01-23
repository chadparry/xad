#!/usr/bin/env python

import cv2
import numpy
import skimage.measure

cap = cv2.VideoCapture('idaho.webm')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))

IMG_WEIGHT = 1/3.

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()
    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sub = fgbg.apply(frame)
    #score = skimage.measure.structural_similarity(next, prev)

    opening = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)
    blurred = cv2.blur(opening, (5,5))

    gbr = numpy.zeros_like(frame)
    gbr[...,1] = blurred
    gbr[...,2] = blurred

    composite = cv2.addWeighted(frame, IMG_WEIGHT, gbr, 1 - IMG_WEIGHT, 0)

    cv2.imshow('frame', composite)
    writer.write(composite)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

    prev = next

cap.release()
writer.release()
cv2.destroyAllWindows()
