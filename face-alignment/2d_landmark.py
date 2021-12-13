import face_alignment
from skimage import io
import numpy as np
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device='cuda:0',face_detector='blazeface') #face_detector: sfd,blazeface,dlib

input = io.imread('I2G\man7.jpg')
preds = fa.get_landmarks(input)
## preds = fa.get_landmarks_from_directory('Dirname')
print(preds)
np.save('I2G/man7.npy',np.squeeze(preds))