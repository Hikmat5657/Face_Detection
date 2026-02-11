import cv2
import matplotlib.pyplot as plt 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')# downloads the pre trained model to detect face
img = cv2.imread('face.jpeg') #load the picture
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = face_cascade.detectMultiScale(img, 1.3, 5)# detects faces of different sizes in the input image
print(faces) # faces will output a coordinate
for (x,y,w,h) in faces:
    print(x)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2) #draw a rectangle based on the rectangle
    plt.imshow(img)
    plt.show()
    