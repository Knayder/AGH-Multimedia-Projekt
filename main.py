"""
import numpy as np
import cv2

cap = cv2.VideoCapture('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

if __name__ == '__main__' :
    # Read image
    img = cv2.imread("test.png")
    cv2.namedWindow("Image",2)
    roi = cv2.selectROI("Image", img, False, False)

    ## Display the roi
    if roi is not None:
        x,y,w,h = roi
        mask = np.zeros_like(img, np.uint8)
        cv2.rectangle(mask, (x,y), (x+w, y+h), (255,255,255), -1)
        masked = cv2.bitwise_and(img, mask )
        cv2.imshow("mask", mask)
        cv2.imshow("ROI", masked)

    cv2.waitKey()

import cv2
import numpy as np

drawing = False
def draw_circle(event,x,y,flags,param):
    #if event == cv2.EVENT_LBUTTONDBLCLK:
    print(cv2.EVENT_MBUTTONUP)
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
    if drawing:
        cv2.circle(img,(x,y),20,(255,0,0),-1)
    

# Create a black image, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
"""
import numpy as np
import cv2
#from skimage.metrics import structural_similarity
import time

#cap = cv2.VideoCapture('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')
cap = cv2.VideoCapture('https://edge01.cdn.wolfcloud.pl/lookcam/6Q3eV9rn8O04xwZGyXWALe1kRDlm6VoPk6zKpMaE3rP5d7BQYv92qgbJnRyZ2g5a/playlist.m3u8?token=J8aY7ewuMWIjLFgMlajp-g&expires=1616106248')

# szczerze nie wiem czy potrzebne, albo nie do końca wiem jak działa

# Ustawienie wielkości buffera dla klatek
# cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

# =========================================================================
# Stałe
MIN_AREA = 200

# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Wymiary do skalowania klatek
WIDTH_BLUR = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
HEIGHT_BLUR = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 4)

WIDTH_DETECT = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 10)
HEIGHT_DETECT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 10)

OG_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
OG_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Długość buffera tła
BG_BUFFER_SIZE = 10

MOTION_BUFFER_SIZE = 5

BRIGHTNESS_DISCARD = 20
# =========================================================================

# Code taken from https://gist.github.com/pknowledge/623515e8ab35f1771ca2186630a13d14
# and https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# https://github.com/methylDragon/opencv-motion-detector/blob/master/Motion%20Detector.py and
# Read two frames one after another
# First value in the tuple is 'true', so we ignore it
_, frameOld = cap.read()
_, frameNew = cap.read()

motion_buffer = []
bg_buffer = []

delay_counter = 0



cv2.namedWindow('Security Feed')

drawing = False
def draw_circle(event,x,y,flags,param):
    global drawing
    global maskImg
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
    if drawing:
        cv2.circle(maskImg,(x,y),40,(0,0,0),-1)

maskImg = np.ones((  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3  ), np.uint8)
redBackground = np.ones((  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3  ), np.uint8)
whiteBackground = np.ones((  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3  ), np.uint8)
for y in range(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))):
    for x in range(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))):
        redBackground[y][x] = [255, 255, 0];


cv2.setMouseCallback('Security Feed',draw_circle)

while cap.isOpened():
    # to trzeba będzie po testować ale wydaje mi się że sprawdzanie różnicy już zblurowanych klatek lepiej niweluje
    # szumy, które bardzo lubią być wykrywane jako ruch
    frameOldB = cv2.GaussianBlur(cv2.resize(frameOld*maskImg, (WIDTH_BLUR, HEIGHT_BLUR), interpolation=cv2.INTER_CUBIC), (5, 5), 0)
    frameNewB = cv2.GaussianBlur(cv2.resize(frameOld*maskImg, (WIDTH_BLUR, HEIGHT_BLUR), interpolation=cv2.INTER_CUBIC), (5, 5), 0)

    # Dodajemy nową klatke do bufferów
    motion_buffer.append(frameNewB.astype('float32'))
    bg_buffer.append(frameNewB.astype('float32'))

    if len(motion_buffer) > MOTION_BUFFER_SIZE:
        motion_buffer.pop(0)

    if len(bg_buffer) > BG_BUFFER_SIZE:
        bg_buffer.pop(0)

    avg_bg = np.mean(bg_buffer, axis=0)
    avg_motion = np.mean(motion_buffer, axis=0)

    avg_bg[avg_bg > 254] = 255

    diff = cv2.absdiff(avg_bg, avg_motion)
    diff[diff < BRIGHTNESS_DISCARD] = 0
    diff[diff > 0] = 254

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff = cv2.resize(gray, (WIDTH_DETECT, HEIGHT_DETECT))

    delay_counter += 1

    # Zmieniamy klatkę do porównywania co 5 klatek żeby wykrywać powolny ruch
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        frameOld = frameNew

    # ogolnie to thresholding na samej szarości jest do kitu i czytałem ze powinno sie go robic na hsv tylko wtedy
    # moze pojawic sie taki problem ze bedzie to zbyt czule u znowu szumy beda wykrywane jako ruch
    # znalazlem nawet kod na stacku
    # https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
    # tutaj też ciekawe rzeczy SPRAWDZONE XD skutecznie obniża fpsy do 5
    # https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python

    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(thresh, None, iterations=4)

    dilated = dilated.astype(np.uint8)
    dilated = cv2.resize(dilated, (OG_WIDTH, OG_HEIGHT))

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        # Rysowanie prostokątów wokół ruszających się obiektóœ
        (x, y, width, height) = cv2.boundingRect(contour)
        cv2.rectangle(frameOld, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frameOld, "Movement detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    cv2.imshow("Security Feed", frameOld*(whiteBackground + redBackground*(1-maskImg)))
    #cv2.imshow("Security Feed", 255*maskImg)
    #cv2.imshow("Security Feed", gray);

    frameOld = frameNew
    _, frameNew = cap.read()

    # nie potrzebne aż 40ms czekania, obraz wtedy laguje jeszcze bardziej niż normalnie
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()