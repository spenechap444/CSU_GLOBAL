import cv2
import numpy as np
from PIL import Image
import dlib
import os
import matplotlib.pyplot as plt

class FaceID:
    def __init__(self, img):
        self.img = img
        self.imgd = img
        
    
    def ID_face(self, draw=True):
        #loading the dlib detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(os.path.join('/Users/spencerchapman/Documents/CSU_GLOBAL/CSC515/','shape_predictor_5_face_landmarks.dat'))
        
        #converting to image to gray
        gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        
        #dlib face detection
        detector = dlib.get_frontal_face_detector()
        detections = detector(gray, 1)
        
        # Find landmarks
        sp = dlib.shape_predictor(os.path.join('/Users/spencerchapman/Documents/CSU_GLOBAL/CSC515/','shape_predictor_68_face_landmarks.dat'))
        faces = dlib.full_object_detections()
        for det in detections:
            faces.append(sp(self.img, det))
            #det represents coordinates of the image
        
        #bounding box coordinates
        bb = [i.rect for i in faces]
        bb = [((i.left(), i.top()),
           (i.right(), i.bottom())) for i in bb]
        
        if draw:
            self.draw_face(bb)
        
        return faces

    def ID_eyes(self, faces, id_type='both', draw=True):
        right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
        right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]  # Convert out of dlib format
        left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
        left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]    # Convert out of dlib format
        
        if draw:
            self.draw_eyes(right_eyes, left_eyes, id_type)
        
    def draw_face(self, bb):
        [cv2.rectangle(self.imgd, i[0], i[1], (255, 0, 0), 5) for i in bb]
                
    def draw_eyes(self, right, left, id_type):
        #Need to redesign eyes, currently contains old method
        
        if id_type == 'points' or id_type == 'both':
            for eye in right:
                [cv2.circle(self.imgd, (point[0], point[1]), 1, (0, 255, 0), -1) for point in eye]
            for eye in left:
                [cv2.circle(self.imgd, (point[0], point[1]), 1, (0, 255, 0), -1) for point in eye]
        
        if id_type == 'box' or id_type == 'both':
            for eye in right:
                cv2.rectangle(self.imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                                (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                                (0, 0, 255), 1)
            for eye in left:
                cv2.rectangle(self.imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                                (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                                (0, 0, 255), 1)

        
        

def main():
    img_path = '/Users/spencerchapman/Google Drive/My Drive/CSU_GLOBAL/CSC580/Module_1/mugshot2.jpg'
    img2_path = '/Users/spencerchapman/Google Drive/My Drive/CSU_GLOBAL/CSC580/Module_1/group_pic.png'
    img3_path = '/Users/spencerchapman/Google Drive/My Drive/CSU_GLOBAL/CSC580/Module_1/mugshot1.jpg'
    init_img = cv2.cvtColor(cv2.imread(img3_path), cv2.COLOR_BGR2RGB)
    img = FaceID(init_img)
    faces = img.ID_face()
    img.ID_eyes(faces, 'both', False)
    plt.imshow(img.imgd)

main()
