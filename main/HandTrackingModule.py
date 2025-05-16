import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        if not hasattr(self, 'lmList') or len(self.lmList) == 0:
            return []  # Return an empty list if lmList is not initialized or empty
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)


        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
# import cv2
# import numpy as np
# import math

# class HandTrackerCV:
#     def __init__(self, cascade_path='hand.xml'):
#         self.hand_cascade = cv2.CascadeClassifier(cascade_path)

#     def findHand(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         hands = self.hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#         if len(hands) == 0:
#             return None, None
#         # Assuming the largest detected region is the hand
#         x, y, w, h = max(hands, key=lambda rect: rect[2] * rect[3])
#         hand_roi = frame[y:y+h, x:x+w]
#         return hand_roi, (x, y, w, h)

#     def is_peace_sign(self, hand_roi):
#         # Convert to HSV and create a skin color mask
#         hsv = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2HSV)
#         lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#         upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#         mask = cv2.inRange(hsv, lower_skin, upper_skin)

#         # Apply morphological operations to filter out the background noise
#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=4)
#         mask = cv2.GaussianBlur(mask, (5, 5), 100)

#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         if not contours:
#             return False

#         # Find the largest contour
#         contour = max(contours, key=lambda x: cv2.contourArea(x))
#         hull = cv2.convexHull(contour, returnPoints=False)
#         if hull is None or len(hull) < 3:
#             return False

#         defects = cv2.convexityDefects(contour, hull)
#         if defects is None:
#             return False

#         # Count the number of defects which correspond to fingers
#         finger_count = 0
#         for i in range(defects.shape[0]):
#             s, e, f, d = defects[i, 0]
#             start = tuple(contour[s][0])
#             end = tuple(contour[e][0])
#             far = tuple(contour[f][0])

#             # Calculate the angle between the fingers
#             a = math.dist(end, start)
#             b = math.dist(far, start)
#             c = math.dist(end, far)
#             angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) if b != 0 and c != 0 else 0

#             if angle <= math.pi / 2:
#                 finger_count += 1

#         # If two fingers are up, it's likely a peace sign
#         return finger_count == 2
