import cv2 
import mediapipe as mp 
from math import hypot 
import numpy as np 
from google.protobuf.json_format import MessageToDict
import subprocess
import time

def set_volume_macos(volume_level):
    """Set volume on macOS using osascript"""
    volume = max(0, min(100, volume_level))  # Ensure volume is between 0 and 100
    cmd = f'osascript -e "set volume output volume {volume}"'
    subprocess.run(cmd, shell=True)

def is_palm_open(hand_landmarks):
    """Check if all fingers are extended (palm is open)"""
    fingers_extended = 0
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]
    
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            fingers_extended += 1
            
    if hand_landmarks[4].x < hand_landmarks[3].x:  # For left hand
        fingers_extended += 1
    
    print(f"Fingers Extended: {fingers_extended}")    
    return fingers_extended >= 4


cap = cv2.VideoCapture(0)


mpHands = mp.solutions.hands 
hands = mpHands.Hands(min_detection_confidence=0.75)
mpDraw = mp.solutions.drawing_utils


current_volume = None
is_volume_locked = False
palm_start_time = None
PALM_HOLD_TIME = 2.0
palm_progress = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break
        
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    left_lmList, right_lmList = [], []
    palm_detected = False
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = MessageToDict(hand_handedness)['classification'][0]['label']
            hand_landmarks = results.multi_hand_landmarks[idx]
            
            if label == 'Left':
                #Checking for palm guesters
                if is_palm_open(hand_landmarks.landmark):
                    palm_detected = True
                    if palm_start_time is None:
                        palm_start_time = time.time()
                        print("Palm detection started")
                    else:
                        hold_duration = time.time() - palm_start_time
                        palm_progress = min(hold_duration / PALM_HOLD_TIME * 100, 100)
                        print(f"Palm Hold Duration: {hold_duration:.2f}s, Progress: {palm_progress:.1f}%")
                        
                        if hold_duration >= PALM_HOLD_TIME:
                            is_volume_locked = not is_volume_locked
                            print(f"Volume {'Locked' if is_volume_locked else 'Unlocked'}")
                            palm_start_time = None
                            palm_progress = 0
                
                for lm in hand_landmarks.landmark:
                    h, w, _ = img.shape
                    left_lmList.append([int(lm.x*w), int(lm.y*h)])
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)
                
            if label == 'Right':
                for lm in hand_landmarks.landmark:
                    h, w, _ = img.shape
                    right_lmList.append([int(lm.x*w), int(lm.y*h)])
                mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    if not palm_detected and palm_start_time is not None:
        print("Palm detection reset")
        palm_start_time = None
        palm_progress = 0

    # Right hand controls 
    if right_lmList and not is_volume_locked:
        x1, y1 = right_lmList[4][0], right_lmList[4][1]  # Thumb tip
        x2, y2 = right_lmList[8][0], right_lmList[8][1]  # Index finger tip
        
        cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 3)
        length = hypot(x2-x1, y2-y1)
        
        #  distance  to volume
        current_volume = np.interp(length, [15,200], [0,100])
        print(f"Finger Distance: {length:.1f} pixels, Volume: {current_volume:.1f}%")
        
        try:
            set_volume_macos(int(current_volume))
        except Exception as e:
            print(f"Failed to set volume: {e}")

    #  status and volume information
    status_color = (0, 0, 255) if is_volume_locked else (0, 255, 0)
    status_text = "LOCKED" if is_volume_locked else "UNLOCKED"
    cv2.putText(img, f"Status: {status_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    if current_volume is not None:
        cv2.putText(img, f"Volume: {current_volume:.1f}%", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


    if palm_progress > 0:
        bar_width = 200
        filled_width = int((palm_progress / 100) * bar_width)
        cv2.rectangle(img, (10, 100), (10 + bar_width, 120), (255,255,255), 2)
        cv2.rectangle(img, (10, 100), (10 + filled_width, 120), (0,255,0), cv2.FILLED)
        cv2.putText(img, f"Hold Progress: {palm_progress:.0f}%", 
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


    cv2.putText(img, "Hold left palm for 2s: Lock/Unlock", (10, img.shape[0]-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(img, "Right hand: Control volume", (10, img.shape[0]-30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
    cv2.imshow('Volume Control', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()