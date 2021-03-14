import numpy as np
import cv2
import time

cap = cv2.VideoCapture('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
# FPS = 1/X
# X = desired FPS
MIN_AREA = 900
FPS = 1/30
FPS_MS = int(FPS * 1000)

# Code taken from https://gist.github.com/pknowledge/623515e8ab35f1771ca2186630a13d14
# and https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

while cap.isOpened():
    # Read two frames one after another
    # First value in the tuple is 'true', so we ignore it
    _, frameOld = cap.read()
    _, frameNew = cap.read()

    # Convert the first frame to greyscale
    frameOldGreyscale = cv2.cvtColor(frameOld, cv2.COLOR_BGR2GRAY)
    # Convert the second frame to greyscale and blur it
    frameNewGreyscale = cv2.cvtColor(frameNew, cv2.COLOR_BGR2GRAY)

    # Difference between frames
    diff = cv2.absdiff(frameOldGreyscale, frameNewGreyscale)

    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        (x, y, width, height) = cv2.boundingRect(contour)
        cv2.rectangle(frameOld, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frameOld, "Movement detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    cv2.imshow("Security Feed", frameOld)
    # cv2.imshow("Thresh", thresh)
    # cv2.imshow("Frame Delta", diff)

    frameOld = frameNew
    _, frameNew = cap.read()

    if cv2.waitKey(40) == 27:
        break

    #time.sleep(FPS)
    cv2.waitKey(FPS_MS)

cap.release()
cv2.destroyAllWindows()
