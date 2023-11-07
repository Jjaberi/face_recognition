import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

           
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
      
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
           
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

          
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

   
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names




----------------------------------------------------------
This code defines a class SimpleFacerec for face recognition. Here's a breakdown of what each part does:

__init__(self): This is the constructor method that's called when you create a new instance of the class. It initializes the instance variables known_face_encodings and known_face_names as empty lists, and sets frame_resizing to 0.25.

load_encoding_images(self, images_path): This is a method that takes a path to a directory of images as an argument. It uses the glob and os modules to find all files in the specified directory, regardless of file type. It then prints the number of images found.

This block of code is part of the load_encoding_images method in the SimpleFacerec class. It's iterating over each image path in the images_path list. Here's what each line does:

img = cv2.imread(img_path): This line reads the image file at the given path.

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB): This line converts the image from BGR color space (which is the default in OpenCV) to RGB color space (which is used by the face_recognition library).

basename = os.path.basename(img_path): This line gets the base name of the image file (i.e., the file name without the directory path).

(filename, ext) = os.path.splitext(basename): This line splits the base name into the file name and the extension.

img_encoding = face_recognition.face_encodings(rgb_img)[0]: This line uses the face_recognition library to generate a face encoding for the image. It assumes that there's exactly one face in the image and takes the encoding of that face.

self.known_face_encodings.append(img_encoding): This line adds the face encoding to the list of known face encodings.

self.known_face_names.append(filename): This line adds the file name (which presumably is the name of the person in the image) to the list of known face names.

print("Encoding images loaded"): This line prints a message indicating that the images have been loaded and encoded.

This code defines a method detect_known_faces in the SimpleFacerec class. Here's what each part does:

small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing): This line resizes the input frame to a smaller size for faster face recognition.

rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB): This line converts the small frame from BGR to RGB color space.

face_locations = face_recognition.face_locations(rgb_small_frame): This line detects the locations of faces in the small frame.

face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations): This line generates face encodings for the detected faces.

face_names = []: This line initializes an empty list to store the names of the people in the frame.

The for loop iterates over each face encoding. For each encoding, it compares it with the known face encodings to see if it matches any of them. If it does, it assigns the corresponding name to the face; otherwise, it assigns the name "Unknown".
This code completes the detect_known_faces method:

face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding): This line calculates the face distance between the current face encoding and all known face encodings.

best_match_index = np.argmin(face_distances): This line finds the index of the smallest face distance, which corresponds to the best match.

if matches[best_match_index]: name = self.known_face_names[best_match_index]: If the best match is a valid match, the name of the face is set to the corresponding known face name.

face_names.append(name): The determined name is added to the face_names list.

face_locations = np.array(face_locations): The face locations are converted to a numpy array.

face_locations = face_locations / self.frame_resizing: The face locations are scaled back to the original frame size.

return face_locations.astype(int), face_names: The method returns the face locations and names.






























