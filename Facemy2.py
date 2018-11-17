import cv2
import numpy as np
import os
import pandas as pd
import take_pictures

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#subjects = ["", "Mandy", "Shushi", "Rishav", "Taran", "Ravi"]
subjects_file_path = "C:/Users/HP PRO/Desktop/python/Facedetect/subjects.csv"
subjects_csv = pd.read_csv(subjects_file_path)
#subjects = subject_csv.tolist()
subjects = subjects_csv.columns.tolist()

path = "C:/Users/HP PRO/Desktop/training-data/"

def if_new(path):
    label = raw_input("Enter your name: ")
    #subjects.append(label)
    lengthdir = len(os.listdir(path))
    lengthdir = lengthdir + 1
    newpath = path+'s'+str(lengthdir)
    os.mkdir(newpath)
    f = open(subjects_file_path, 'a')
    f.write(","+label)
    take_pictures.tp(newpath)

def detect_face(img):
    
    #convert the test image to gray image as opencv face detector expects gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #the caascade classifier
    #face_cascade = cv2.CascadeClassifier('C:/Users/HP PRO/Desktop/python/Facedetect/haarcascade_frontalface_default.xml')
    
    #let's detect multiscale images
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    
    #if no face found return none
    if (len(faces)) == 0:
        return None, None
    
    #as we know there is only one face in training images
    (x, y, w, h) = faces[0]
    
    return gray_img[y:y+w, x:x+h], faces[0]

def prepare_training_data(path):
    
    #get the directories in data folder
    dirs = os.listdir(path)
    
    #list to hold faces
    faces = []
    
    #list to hold labels
    labels = []
    
    #read img from each directory
    for dir_name in dirs:
        
        #as our dir name starts with 's'
        if not dir_name.startswith("s"):
            continue
            
        #extract labels for each image
        label = int(dir_name.replace("s", ""))
        
        #build path of direcrtory containing images
        s_dir_path = path + "/" + dir_name
        
        #get the images under above created path
        s_images_name = os.listdir(s_dir_path)
        
        
        #read each image
        for image_name in s_images_name:
            
            #ignore file which starts with '.'
            if image_name.startswith("."):
                continue
            
            #build image path like training-data/s1/1.jpg
            image_path = s_dir_path + "/" + image_name
            
            #read image
            image = cv2.imread(image_path)
            
            #display the image
            #cv2.imshow("Training image..", image)
            cv2.waitKey(100)
            
            #now detect the face in the image
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        
    return faces, labels

def draw_rect(img, x, y, w, h):
    #(x, y, w, h) = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
#function to draw text
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

#now make a prediction function
def predict(test_img,x,y,w,h):
    #predict image using our recognizer
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]
    
    draw_text(img, label_text, x, y)
    
    return img

#if new user?
newuser = raw_input("New User? Enter Y/N : ")
if(newuser == 'Y' or newuser == 'y'):
    if_new(path)

#now let's first prepare our training set
print("Preparing data for model input")
faces, labels = prepare_training_data(path)
print("Data Prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

print("Predicting images")

face_recognizer.train(faces, np.array(labels))

cap=cv2.VideoCapture(0)
while True:
    ret, img=cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        draw_rect(img, x, y, w, h)
        gray_img = gray[y:y+w, x:x+h]
        face = gray_img
        rect = faces[0]
        predicted_img = predict(face,x,y,w,h)
        #roi_gray = gray_img[y:y+h, x:x+w]
        #roi_color = predicted_img[y:y+h, x:x+w]
    cv2.imshow('img', img)
    #else:
    #    cv2.imshow("No Face Found!", img)
    k=cv2.waitKey(30) & 0xFF
    if k==32:
            break 

cap.release()
cv2.destroyAllWindows()
