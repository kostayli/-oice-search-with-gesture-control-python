import cv2
import mediapipe as mp
import time
import math
import webbrowser
import numpy as np
import pyautogui as pgui
import speech_recognition as sr


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHand = mp.solutions.hands
        self.Hand = self.mpHand.Hands( self.mode, self.maxHands, 1, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def detect(self, img, is_draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.Hand.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for Landmarks in self.results.multi_hand_landmarks:
                if is_draw:
                    self.mpDraw.draw_landmarks(img, Landmarks, self.mpHand.HAND_CONNECTIONS)
        return img

    def createList(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)
        return self.lmList
    def findDistace(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255, 0, 255), t)
            cv2.circle(img, (x1,y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2,y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx,cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1,y1, x2,y2,cx,cy]
    def fingersUp(self):
        self.ups = []
        if len(self.lmList)!=0:
            if self.lmList[4][1]>self.lmList[3][1]:
                self.ups.append(1)
            else:
                self.ups.append(0)

            if self.lmList[8][2] < self.lmList[5][2]:
                self.ups.append(1)
            else:
                self.ups.append(0)

            if self.lmList[12][2] < self.lmList[9][2]:
                self.ups.append(1)
            else:
                self.ups.append(0)

            if self.lmList[16][2] < self.lmList[13][2]:
                self.ups.append(1)
            else:
                self.ups.append(0)

            if self.lmList[20][2] < self.lmList[17][2]:
                self.ups.append(1)
            else:
                self.ups.append(0)
        return self.ups

def main():
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    wCam = 640
    hCam = 480
    wScr, hScr = 1920, 1080
    frameRightW = 100
    frameLeftW = 100
    frameUpH = 100
    frameDownH = 100
    smooth = 6
    rclick_time = time.time()
    is_grab =False
    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4, hCam)
    Det = handDetector()
    while True:
        success, img = cap.read()
        img = Det.detect(img)
        List = Det.createList(img)
        if len(List)!=0:
            x1, y1 = List[8][1:]
            Ups = Det.fingersUp()
            cv2.rectangle(img, (frameLeftW, frameDownH), (wCam - frameRightW, hCam - frameUpH), (0, 255, 0))

            if Ups == [1, 1, 0, 0, 0]:
                if(time.time() - rclick_time)>5:
                    r = sr.Recognizer()
                    mic = sr.Microphone()
                    with mic as source:
                        #r.adjust_for_ambient_noise(source)
                       audio = r.listen(source)
                    rclick_time = time.time()
                    print(r.recognize_google(audio, language='ru-RU'))
                    url = "https://yandex.ru/search/?lr=10735&text=" + r.recognize_google(audio, language='ru-RU')
                    webbrowser.open_new_tab(url)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("WebcamShow", img)
        if cv2.waitKey(5) & 0xFF == 27:
            break


if __name__ == "__main__":
    main()