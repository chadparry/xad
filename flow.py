#!/usr/bin/env python

import cv2
import collections
import numpy as np
cap = cv2.VideoCapture('idaho.webm')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter("output.riff", fourcc, 25, (1280, 720))

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

flows = collections.deque()

IMG_WEIGHT = 0.25
FLOW_WEIGHT = 0.5

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) * FLOW_WEIGHT
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    flows.appendleft(rgb)
    if len(flows) > 3:
        flows.pop()
    composite = frame2
    composite *= IMG_WEIGHT
    for f in flows:
        #composite = cv2.addWeighted(composite, IMG_WEIGHT, f, 1 - IMG_WEIGHT, 0)
        composite = cv2.bitwise_or(composite, f)

    cv2.imshow('frame2', composite)
    writer.write(composite)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    prvs = next

cap.release()
writer.release()
cv2.destroyAllWindows()
