import cv2
import mediapipe as mp
import time
from tensorflow.keras import models
import numpy as np
import os
import threading
import pyttsx3 
# engine = pyttsx3.init()

# def speak(text):
#     engine.say(text)
#     engine.runAndWait()

# Create a new thread to speak the text



os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = models.load_model('D:\projects\FY_project\hand_recognition\models\\asl_model3.h5',compile=False)
classes = {0:'yes',1:'no',2:'please',3:'good',4:'hello',5:'you',6:'nice to meet you',7:'name',8:'good morning',9:'how are you ?'}#{0:'yes',1:'no',2:'please',3:'good',4:'hello',5:'you'}

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

image_count = 1000
cx=cy=cx_max=cy_max=0
prediction = -1

previous = ''

while True:
    success, img = cap.read()
    img_copy = img[:,:]
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        cx,cy = 100000,100000
        cx_max,cy_max = -100000,-100000
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx = min(cx,int(lm.x *w))
                cy = min(cy,int(lm.y *h))
                cx_max = max(cx_max,int(lm.x *w))
                cy_max = max(cy_max,int(lm.y *h))
        cv2.rectangle(img,(cx-50,cy-50),(cx_max+50,cy_max+50),(0,255,0), 1)
        test_img = img_copy[cy-50:cy_max+50,cx-50:cx_max+50]
        test_img = cv2.resize(cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY),(32,32))/255
        # print(test_img)
        probabilities = model.predict(test_img.reshape(-1,32,32,1))
        print(probabilities)
        prediction=np.argmax(probabilities)

        # if image_count>500:
        #     print(f'({cx},{cy}),({cx_max},{cy_max})')
        #     cv2.imwrite('please/please_'+str(image_count)+'.jpg',img_copy[cy-50:cy_max+50,cx-50:cx_max+50])
        #     image_count-=1

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    # engine.say(classes[prediction])
    # engine.runAndWait()
    # if previous!=classes[prediction]:
    #     previous = classes[prediction]
    #     t = threading.Thread(target=speak, args=(classes[prediction],))
    #     t.start()

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.putText(img,f'{classes[prediction]} {round(probabilities[0][prediction]*100,2)}% confidence', (cx-50,cy-50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)
    k = cv2.waitKey(10)
    if k==27:
        break