import cv2
import dlib
import numpy as np
from collections import OrderedDict
'''
提取convex
'''
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])
 
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
 ("right_eye", (2, 3)),
 ("left_eye", (0, 1)),
 ("nose", (4)),
])

def shape2np(shape,dtype = 'int'):
    coords = np.zeros((shape.num_parts,2),dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # 创建两个copy
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()
    # 设置一些颜色区域
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]
 
    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # 得到每一个点的坐标
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        if name == 'jaw':
            # 用线条连起来
            for l in range(1, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                # 对位置进行连线
                cv2.line(overlay, ptA, ptB, colors[i], 2)
 
        else:
            # 使用cv2.convexHull获得位置的凸包位置
            hull = cv2.convexHull(pts)
            # 使用cv2.drawContours画出轮廓图
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
            
    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    return output

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('I2G\shape_predictor_68_face_landmarks.dat')

img= cv2.imread('I2G\man7.jpg') #h,w,3
h,w = img.shape[:2]
img = cv2.resize(img,(256,256),interpolation= cv2.INTER_AREA)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

dets = detector(gray,1)
for (i,rect) in enumerate(dets):
    shape = predictor(gray,rect)
    shape = shape2np(shape)

overlay = img.copy()
hull = cv2.convexHull(shape)
cv2.drawContours(img, [hull],-1,(0,0,0),-1)
cv2.imshow('Image',img)
cv2.waitKey(0)

np.save('I2G\man7.npy',shape)

'''  单独展示的部分
     for (i,rect) in enumerate(dets):
    shape = predictor(gray,rect)
    shape = shape2np(shape)
    # 第七步：对字典进行循环
    for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
        clone = img.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255), 2)
        # 根据位置画点
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
 
        #第八步： 使用cv2.boundingRect获得脸部轮廓位置信息
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # 第九步：根据位置提取脸部的图片
        roi = img[y:y+h, x:x+w]
        (h, w) = roi.shape[:2]
        width = 250
        r = width / float(w)
        dim = (width, int(r*h))
        roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

        cv2.imshow('ROI', roi)
        cv2.imshow('Image', clone)
        cv2.waitKey(0)
'''