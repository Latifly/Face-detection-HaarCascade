import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Read the input image
img = cv2.imread('test.jpg')

scale_percent = 70 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    roiFace = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roiFace, 1.11, 3)
    smiles = smile_cascade.detectMultiScale(roiFace, 1.1, 120)

    for(x1,y1,w1,h1) in eyes:
        cv2.rectangle(img, (x+x1,y+y1), (x+x1 + w1, y+y1 + h1), (0, 250, 0), 2)

    for(x1,y1,w1,h1) in smiles:
        cv2.rectangle(img, (x+x1,y+y1), (x+x1 + w1, y+y1 + h1), (0, 250, 255), 2)
    
# Display the output
cv2.imshow('img', img)
cv2.waitKey()