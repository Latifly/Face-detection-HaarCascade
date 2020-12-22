from cv2 import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('LB - 07 [480p].mkv')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        roiFace = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roiFace, 1.1, 2)
        smiles = smile_cascade.detectMultiScale(roiFace, 1.5, 27)

        for(x1,y1,w1,h1) in eyes:
            cv2.rectangle(img, (x+x1,y+y1), (x+x1 + w1, y+y1 + h1), (0, 250, 0), 2)

        for(x1,y1,w1,h1) in smiles:
            cv2.rectangle(img, (x+x1,y+y1), (x+x1 + w1, y+y1 + h1), (0, 250, 255), 2)
  
    # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
# Release the VideoCapture object
cap.release()