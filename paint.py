import cv2 as cv
import mediapipe as mp
import math
import numpy as np

cap=cv.VideoCapture(1)
cap.set(3,1280)
cap.set(4,728)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


brushThickness = 25
eraserThickness = 100
drawColor = (255, 0, 255)
########################

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
 
 
overlayList = []

image = cv.imread(r'img1.png')
image= cv.resize(image, (1280,125), interpolation = cv.INTER_AREA)
overlayList.append(image)
image = cv.imread(r'img2.png')
image= cv.resize(image, (1280,125), interpolation = cv.INTER_AREA)
overlayList.append(image)
image = cv.imread(r'img3.png')
image= cv.resize(image, (1280,125), interpolation = cv.INTER_AREA)
overlayList.append(image)
image = cv.imread(r'img4.png')
image= cv.resize(image, (1280,125), interpolation = cv.INTER_AREA)
overlayList.append(image)
#print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)


def fingercount(gg):
    fing=[1,0,0,0,0]
    if gg[8][1]<gg[6][1]:
        fing[1]=1
    if gg[12][1]<gg[10][1]:
        fing[2]=1
    if gg[16][1]<gg[14][1]:
        fing[3]=1
    if gg[20][1]<gg[18][1]:
        fing[4]=1
    return fing

while True:
    success, img = cap.read()
    #img=cv.flip(img,1)
    i1 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(i1)
    #print(results.multi hand landmarks)
    
    #img=cv.flip(img,1)
    img[0:125,0:1280]=header
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            gg=[]
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy= int(lm.x*w), int(lm.y*h)
                gg.append([cx,cy])
                k,l=str(cx),str(cy)
                #print(gg)
                #cv.putText(img, str(k+","+l), (cx, cy),cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
                if len(gg)==21:
                    
                    x1, y1 = gg[8]
                    x2, y2 = gg[12]
                    fing=fingercount(gg)

                    if fing[1] and fing[2]:
                        xp, yp = 0, 0
                        #print("selection mode")
                        if y1 < 125:
                            if 250 < x1 < 450:
                                header = overlayList[0]
                                drawColor = (255, 0, 255)
                            elif 550 < x1 < 750:
                                header = overlayList[1]
                                drawColor = (255, 0, 0)
                            elif 800 < x1 < 950:
                                header = overlayList[2]
                                drawColor = (0, 255, 0)
                            elif 1050 < x1 < 1200:
                                header = overlayList[3]
                                drawColor = (0, 0, 0)
                        cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv.FILLED)

                    if fing[1] and fing[2]==False:
                        cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
                        #print("Drawing Mode")
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
             
                        cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
             
                        if drawColor == (0, 0, 0):
                           cv.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                           cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        
                        else:
                            cv.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                            cv.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
         
                        xp, yp = x1, y1
             
             
                    #Clear Canvas when all fingers are up
                    if all (x >= 1 for x in fing):
                        imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                        

                        
                                #cv.putText(img, "fc="+str(fingercount(gg)), (20, 20),cv.FONT_HERSHEY_COMPLEX, 0.5, 0, 1)
                               
                            #print(id, cx, cy)
                        #print()
            mpDraw.draw_landmarks (img, handLms, mpHands. HAND_CONNECTIONS)
            
    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 20, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img = cv.bitwise_and(img,imgInv)
    img = cv.bitwise_or(img,imgCanvas)
    cv.imshow("ok",img)
    #cv.imshow("Canvas", imgCanvas)
    cv.waitKey(1)
cv.DestroyWindow("ok")
