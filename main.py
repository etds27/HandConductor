#!/usr/bin/env python3

import logging
import cv2
import mediapipe as mp
import HandController
import LightController

# initialize the camera and grab a reference to the raw camera capture

cap = cv2.VideoCapture()
video = cv2.VideoWriter('light_demo.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))
cap.open("http://192.168.0.218:8081")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


lc = LightController.LightController()

prev_up = 0

# For webcam input:
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,  # set to single hand
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hc = HandController.HandController(hand_landmarks)
                image = hc.create_image(image)
                num_fingers_up = hc.get_num_fingers_up()

                if not prev_up == num_fingers_up and num_fingers_up:
                    logging.debug("main: old_fin_count = %i | new_fin_count = %s" % (prev_up, num_fingers_up))
                    logging.info("main: new finger count. Sending command")
                    lc.set_color_by_finger(num_fingers_up)
                    prev_up = num_fingers_up


        # Flip the image horizontally for a selfie-view display.
        # image = cv2.flip(image, 0)
        image = cv2.flip(image, 2)
        cv2.imshow('Light Conductor', cv2.flip(image, 1))
        video.write(image)



        if cv2.waitKey(5) & 0xFF == 27:
            break



cap.release()
video.release()
