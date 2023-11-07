import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("./images")


cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1  = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        
        cv2.putText(frame, name,  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame,(x1, y1), (x2, y2), (0, 0, 200 ), 4)
    
    
    cv2.imshow("Frame",frame)
        
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()



---------------------------------------------------
import cv2: This line imports the OpenCV library, which is used for computer vision tasks.

from simple_facerec import SimpleFacerec: This line imports the SimpleFacerec class from the simple_facerec module.

sfr = SimpleFacerec(): Here, an instance of the SimpleFacerec class is created.

sfr.load_encoding_images("./images"): This line loads images from the "./images" directory to be used for face recognition.

cap = cv2.VideoCapture(0): This line initializes a video capture object to capture video from the webcam.

while True:: This line starts an infinite loop, which will continue until it's manually stopped.

ret, frame = cap.read(): This line captures a frame from the video.

face_locations, face_names = sfr.detect_known_faces(frame): This line uses the detect_known_faces method to detect faces in the frame and identify them.

The for loop iterates over each detected face and its corresponding name.

cv2.putText and cv2.rectangle are used to draw the name and a rectangle around each detected face on the frame.

cv2.imshow("Frame",frame): This line displays the frame with the drawn rectangles and names.

key = cv2.waitKey(1): This line waits for a key press for 1 millisecond.

if key == 27: break: If the 'Esc' key (key code 27) is pressed, the loop is broken and the program ends.

cap.release(): This line releases the video capture object.

cv2.destroyAllWindows(): This line closes all OpenCV windows.