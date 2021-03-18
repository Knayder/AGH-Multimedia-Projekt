import numpy as np
import cv2
import time

# cap = cv2.VideoCapture('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')
cap = cv2.VideoCapture('https://edge01.cdn.wolfcloud.pl/lookcam/6Q3eV9rn8O04xwZGyXWALe1kRDlm6VoPk6zKpMaE3rP5d7BQYv92qgbJnRyZ2g5a/playlist.m3u8?token=J8aY7ewuMWIjLFgMlajp-g&expires=1616106248')

#szczerze nie wiem czy potrzebne, albo nie do końca wiem jak działa
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# FPS = 1/X
# X = desired FPS

#było 900 ale nie wyłapywało ludzi
MIN_AREA = 200

FPS = 1/30
FPS_MS = int(FPS * 1000)

# Code taken from https://gist.github.com/pknowledge/623515e8ab35f1771ca2186630a13d14
# and https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# Read two frames one after another
# First value in the tuple is 'true', so we ignore it
#na koncu pętli robiona podmianka to bez sensu jeszcze raz robić to na początku
_, frameOld = cap.read()
_, frameNew = cap.read()

while cap.isOpened():
    # Convert the first frame to greyscale
    # frameOldGreyscale = cv2.cvtColor(frameOld, cv2.COLOR_BGR2GRAY)
    # Convert the second frame to greyscale and blur it
    # frameNewGreyscale = cv2.cvtColor(frameNew, cv2.COLOR_BGR2GRAY)

    # Difference between frames

    #to trzeba będzie po testować ale wydaje mi się że sprawdzanie różnicy już zblurowanych klatek lepiej niweluje szumy, które bardzo lubią być wykrywane jako ruch
    frameOldB = cv2.GaussianBlur(frameOld, (5, 5), 11)
    frameNewB = cv2.GaussianBlur(frameNew, (5, 5), 11)

    diff = cv2.absdiff(frameOldB, frameNewB)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #był threshold ale brakowało blura
    # blur = cv2.GaussianBlur(gray, (31, 31), 3)
    # thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.threshold(gray, 23, 200, cv2.THRESH_TOZERO)[1]
    dilated = cv2.dilate(thresh, None, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    #nie potrzebne aż 40ms czekania, obraz wtedy laguje jeszcze bardziej niż normalnie
    if cv2.waitKey(1) == 27:
        break

    #time.sleep(FPS)
    #cv2.waitKey(FPS_MS)

cap.release()
cv2.destroyAllWindows()
