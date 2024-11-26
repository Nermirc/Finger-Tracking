import cv2       
import time
import mediapipe as mp
    
    
cap = cv2.VideoCapture(0)  
if not cap.isOpened():
        print("Kamera açılamadı !")
        exit()

    
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

tipIds = [4, 8, 12, 16, 20]  # Parmak uçları için ID'ler
fingers = []  # Parmak durumlarını tutan liste
while True:
    # Görüntüyü RGB formatına çevirme
    success, img = cap.read()
    if not success:
            print("Döngüyü kır")
            break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # El tespiti için mediapipe Hands'i çalıştırma
    results = hands.process(imgRGB)

    # Tespit edilen elleri çizdirme
    lmList = []  # Her döngüde sıfırlanır
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHand.HAND_CONNECTIONS)

            for id, lm in enumerate(handLms.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        if len(lmList) != 0:
            fingers = []
            # baş parmak
            if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                #4 parmak
            for id in range(1, 5):
                # Parmak uçlarının bir önceki ekleme göre yukarıda olup olmadığını kontrol et
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)  # Yukarıda
                else:
                    fingers.append(0)  # Aşağıda

            totalF = fingers.count(1)
            print(totalF)    

        cv2.putText(img , str(totalF),(30,150),cv2.FONT_ITALIC,2,(255,0,0),5)
    # Görüntüyü göster
    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çıkış
        break

cap.release()
cv2.destroyAllWindows()    
