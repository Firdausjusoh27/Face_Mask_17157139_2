import numpy as np
import cv2
import random

eye_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_eye.xml')
upper_body = cv2.CascadeClassifier('data\\xml\\haarcascade_upperbody.xml')
face_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_frontalface_default.xml')
mouth_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('data\\xml\\haarcascade_nose.xml')

# This is edited version 14/1/2020
# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
no_face_found = (0, 0, 255)
weared_mask_font_color = (124, 252, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"
wear_properly = "Please wear Mask Properly"

# Read video of user using front camera of computer
cap = cv2.VideoCapture(0)

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img, 1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if (len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, no_face_found, thickness, cv2.LINE_AA)
    elif (len(faces) == 0 and len(faces_bw) == 1):
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect lips counters
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.5, 5)
            nose = nose_cascade.detectMultiScale(roi_color, 2, 5)
            print("nose", len(nose))
            if len(nose) == 1:
                for (ex, ey, ew, eh) in nose:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Face detected but Lips not detected which means person is wearing mask
        if len(mouth_rects) == 0:
            print(len(mouth_rects))
            cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        else:
            # lips is detected
            if len(mouth_rects) > 1:
                print(len(mouth_rects))
                for (mx, my, mw, mh) in mouth_rects:

                    if y < my < y + h:
                        # Face and Lips are detected but lips coordinates are within face cordinates which `means lips prediction is true and
                        # person is not waring mask
                        cv2.putText(img, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness,
                                    cv2.LINE_AA)
                        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 0, 255), 3)
                        break

    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
