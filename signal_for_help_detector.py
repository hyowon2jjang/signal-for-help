import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

touch_count = 0
alarm_time = 0
alarm = False
finger_on = False

cap = cv2.VideoCapture(0)

def cal_hand_size(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    wrist_coords = np.array([wrist.x, wrist.y])
    index_mcp_coords = np.array([index_mcp.x, index_mcp.y])
    ring_mcp_coords = np.array([ring_mcp.x, ring_mcp.y])

    return max(np.linalg.norm(wrist_coords - index_mcp_coords), 
               np.linalg.norm(wrist_coords - ring_mcp_coords), 
               np.linalg.norm(index_mcp_coords - ring_mcp_coords),)


def is_thumb_closed(hand_landmarks, hand_size):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ring_finger_root = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    thumb_tip_coords = np.array([thumb_tip.x, thumb_tip.y])
    index_tip_coords = np.array([ring_finger_root.x, ring_finger_root.y])

    distance = np.linalg.norm(thumb_tip_coords - index_tip_coords)

    threshold = 0.15 * hand_size  

    return distance < threshold


def is_others_closed(hand_landmarks, hand_size):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    wrist_coords = np.array([wrist.x, wrist.y])
    index_tip_coords = np.array([index_tip.x, index_tip.y])
    middle_tip_coords = np.array([middle_tip.x, middle_tip.y])
    ring_tip_coords = np.array([ring_tip.x, ring_tip.y])
    pinky_tip_coords = np.array([pinky_tip.x, pinky_tip.y])

    distance_1 = np.linalg.norm(wrist_coords - index_tip_coords)
    distance_2 = np.linalg.norm(wrist_coords - middle_tip_coords)
    distance_3 = np.linalg.norm(wrist_coords - ring_tip_coords)
    distance_4 = np.linalg.norm(wrist_coords - pinky_tip_coords)

    max_distance = max(distance_1, distance_2, distance_3, distance_4)

    threshold = 1 * hand_size 

    return max_distance < threshold


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    touching = False
    if touch_count > 0:
        touch_count -= 1
        if is_others_closed(hand_landmarks, hand_size):
            alarm = True

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if finger_on:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_size = cal_hand_size(hand_landmarks)
            if is_thumb_closed(hand_landmarks, hand_size):
                touching = True
                touch_count = 10
                
    if alarm:
        alarm_time += 1

    if alarm_time >= 60:
        alarm = False
        alarm_time = 0

    if alarm and alarm_time % 20 < 10:
        red_overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
        frame = cv2.addWeighted(frame, 0.5, red_overlay, 0.5, 0)

    cv2.imshow('Signal for help dtection', frame)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        finger_on = not finger_on

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
