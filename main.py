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

# Dumpster
# TODO - REMOVE!
# =========================================================================
# cap = cv2.VideoCapture('https://edge01.cdn.wolfcloud.pl/lookcam/6Q3eV9rn8O04xwZGyXWALe1kRDlm6VoPk6zKpMaE3rP5d7BQYv92qgbJnRyZ2g5a/playlist.m3u8?token=J8aY7ewuMWIjLFgMlajp-g&expires=1616106248')
# cap = cv2.VideoCapture('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')
# Code taken from https://gist.github.com/pknowledge/623515e8ab35f1771ca2186630a13d14
# and https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# https://github.com/methylDragon/opencv-motion-detector/blob/master/Motion%20Detector.py and
# ========================================================================

# =========================================================================
# Constants
MIN_AREA = 200
BRIGHTNESS_DISCARD = 20
# =========================================================================

# Options
# ==========================================================================
cv2.namedWindow('Security Feed')


# ==========================================================================


class imageProcessor:
    def __init__(self, blur_str, blur_width, blur_height, discard, widthDetect, heightDetect):
        self.BLUR_STR = blur_str
        self.BLUR_WIDTH = blur_width
        self.BLUR_HEIGHT = blur_height
        self.BRIGHTNESS_DISCARD = discard
        self.WIDTH_DETECT = widthDetect
        self.HEIGHT_DETECT = heightDetect

    def preprocess_frame(self, frame):
        return cv2.GaussianBlur(cv2.resize(frame, (self.BLUR_WIDTH, self.BLUR_HEIGHT), interpolation=cv2.INTER_CUBIC),
                                (self.BLUR_STR, self.BLUR_STR), 0)

    def calculate_diff(self, frame1, frame2):
        diff = cv2.absdiff(frame1, frame2)
        diff[diff < BRIGHTNESS_DISCARD] = 0
        diff[diff > 0] = 254

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Możliwe że nie zadziała (oryginalny kod troszeczkę inny)
        return cv2.resize(gray, (self.WIDTH_DETECT, self.HEIGHT_DETECT))


class detector:
    def __init__(self, cap_path):
        self.cap = cv2.VideoCapture(cap_path)

        # Buffers
        self.MOTION_BUFFER_SIZE = 10
        self.BG_BUFFER_SIZE = 5

        # Original size of the frame
        self.OG_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.OG_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.maskImg = np.ones(
            (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), np.uint8)
        self.redBackground = np.ones(
            (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
            np.uint8)
        self.whiteBackground = np.ones(
            (int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3),
            np.uint8)

        for y in range(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))):
            for x in range(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))):
                self.redBackground[y][x] = [255, 255, 0]

        # For drawing the mask on top of the frame
        self.drawing = False
        self.erasing = False

        # For image processing
        self.processor = imageProcessor(5, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 4), 20,
                                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 10),
                                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 10))

        self.DEBUG_MODE = False

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
        if self.drawing:
            cv2.circle(self.maskImg, (x, y), 40, (0, 0, 0), -1)
            cv2.circle(self.redBackground, (x, y), 40, (255, 255, 0), -1)
        if event == cv2.EVENT_RBUTTONDOWN:
            self.erasing = True
        if event == cv2.EVENT_RBUTTONUP:
            self.erasing = False
        if self.erasing:
            cv2.circle(self.redBackground, (x, y), 40, (0, 0, 0), -1)
            cv2.circle(self.maskImg, (x, y), 40, (255, 255, 255), -1)

    def detect(self):
        # Read two frames one after another
        # First value in the tuple is 'true', so we ignore it
        _, frame_old = self.cap.read()
        _, frame_new = self.cap.read()

        motion_buffer = []
        bg_buffer = []

        cv2.setMouseCallback('Security Feed', self.draw_circle)

        while self.cap.isOpened():
            frame_old_b = self.processor.preprocess_frame(frame_old * self.maskImg)
            frame_new_b = self.processor.preprocess_frame(frame_new * self.maskImg)

            # Add new frames to buffer
            motion_buffer.append(frame_new_b.astype('float32'))
            bg_buffer.append(frame_new_b.astype('float32'))

            if len(motion_buffer) > self.MOTION_BUFFER_SIZE:
                motion_buffer.pop(0)

            if len(bg_buffer) > self.BG_BUFFER_SIZE:
                bg_buffer.pop(0)

            avg_bg = np.mean(bg_buffer, axis=0)
            avg_motion = np.mean(motion_buffer, axis=0)

            avg_bg[avg_bg > 254] = 255

            gray = self.processor.calculate_diff(avg_bg, avg_motion)

            thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
            dilated = cv2.dilate(thresh, None, iterations=4)

            dilated = dilated.astype(np.uint8)
            dilated = cv2.resize(dilated, (self.OG_WIDTH, self.OG_HEIGHT))

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not self.drawing and not self.erasing:
                # Checking contours for movement
                for contour in contours:
                    if cv2.contourArea(contour) < MIN_AREA:
                        continue

                    # Rectangles drawing
                    (x, y, width, height) = cv2.boundingRect(contour)
                    cv2.rectangle(frame_old, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame_old, "Movement detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 3)

            cv2.imshow("Security Feed", frame_old * (self.whiteBackground + self.redBackground * (1 - self.maskImg)))
            if self.DEBUG_MODE:
                cv2.imshow("Mask", 255 * self.maskImg)
                cv2.imshow("Diff", gray)

            frame_old = frame_new
            _, frame_new = self.cap.read()

            key = cv2.waitKey(10)
            if key == 27:
                break
            elif key == 100:
                if not self.DEBUG_MODE:
                    self.DEBUG_MODE = True
                else:
                    self.DEBUG_MODE = False

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    detector = detector('http://live.uci.agh.edu.pl/video/stream1.cgi?start=1543408695')
    detector.detect()
