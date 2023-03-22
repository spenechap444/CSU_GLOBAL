import cv2
import numpy as np
from PIL import Image
import dlib
import os

img2_path = '/Users/spencerchapman/Documents/CSU_GLOBAL/CSC515/mugshot2.jpg'
init_img = cv2.imread(img2_path)

img = init_img[:, :int(init_img.shape[1]/2), :]
    
    
#loading the dlib detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join('/Users/spencerchapman/Documents/CSU_GLOBAL/CSC515/','shape_predictor_5_face_landmarks.dat'))

#casting image as gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#detecting the face
rects = detector(gray, 0)
#if face found, retrieve coordinates of face rectangle and the shape of our facial landmarks
#facial landmarks are the eyes and nose coordinates
if len(rects) > 0:
    for rect in rects:
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        shape = predictor(gray, rect)
    
def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x,
                                shape.part(i).y)))
    return shape_normal

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eye_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eye_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)

#now we can obtain the central coordinates of nose and eyes
shape = shape_to_normal(shape)
nose, left_eye, right_eye = get_eyes_nose_dlib(shape)

#to find the center o the line between two eyes (endpoint of the median) using this formula
#M = ((x1+x2)/2, (y1+y2)/2)
center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
                      
#Using dlib we had already obtained the coordinates of our face rectangle, now let's find the center of its top side of the face rectangle
center_pred = (int((x+w) / 2), int((y+y)/ 2))

#next is to find the median of the first triangle along with the other two sides of the second triangle
def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

#length1 is the median
#length2 = is the lines between to find the angle
length_line1 = distance(center_of_forehead, nose)
length_line2 = distance(center_pred, nose)
length_line3 = distance(center_pred, center_of_forehead)

#in order to find the angle between two sides of a triangle, knowing three of them
#we can use a formula from a cosine rule
#cosA = (b^2 + c^2 - a^2)/2bc
def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
        
    return cos_a

cos_a = cosine_formula(length_line1, length_line2, length_line3)
angle = np.arccos(cos_a)

#In order to understand what the final angle we will use to rotate our image is
#We need to rotate the endpoint of a median and check if it belongs to the space of the second triangle

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False
    
#the function rotate_point rotates point by an angle around the origin point
#function is_between when given three tops of the triangle and one extra_point checks if the extra point lines in the space of the triangle
rotated_point = rotate_point(nose, center_of_forehead, angle)
rotated_point = (int(rotated_point[0]), int(rotated_point[1]))
if is_between(nose, center_of_forehead, center_pred, rotated_point):
    angle = np.degrees(-angle)
else:
    angle = np.degrees(angle)
    
img = Image.fromarray(img)
img = np.array(img.rotate(angle))

# Convert to dlib
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# dlib face detection
detector = dlib.get_frontal_face_detector()
detections = detector(gray, 1)

# Find landmarks
sp = dlib.shape_predictor(os.path.join('/Users/spencerchapman/Documents/CSU_GLOBAL/CSC515/','shape_predictor_68_face_landmarks.dat'))
faces = dlib.full_object_detections()
for det in detections:
    faces.append(sp(img, det))

# Bounding box and eyes
bb = [i.rect for i in faces]
bb = [((i.left(), i.top()),
       (i.right(), i.bottom())) for i in bb]                            # Convert out of dlib format

right_eyes = [[face.part(i) for i in range(36, 42)] for face in faces]
right_eyes = [[(i.x, i.y) for i in eye] for eye in right_eyes]          # Convert out of dlib format

left_eyes = [[face.part(i) for i in range(42, 48)] for face in faces]
left_eyes = [[(i.x, i.y) for i in eye] for eye in left_eyes]            # Convert out of dlib format

# Display
imgd = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



for i in bb:
    cv2.rectangle(imgd, i[0], i[1], (255, 0, 0), 5)     # Bounding box

for eye in right_eyes:
    cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                        (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                        (0, 0, 255), 5)
    for point in eye:
        cv2.circle(imgd, (point[0], point[1]), 2, (0, 255, 0), -1)

for eye in left_eyes:
    cv2.rectangle(imgd, (max(eye, key=lambda x: x[0])[0], max(eye, key=lambda x: x[1])[1]),
                        (min(eye, key=lambda x: x[0])[0], min(eye, key=lambda x: x[1])[1]),
                        (0, 255, 0), 5)
    for point in eye:
        cv2.circle(imgd, (point[0], point[1]), 2, (0, 0, 255), -1)

cv2.imshow('Rotated img', imgd)
cv2.waitKey(0)
cv2.destoryAllWindows()
