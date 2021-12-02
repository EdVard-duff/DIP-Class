import cv2
import dlib
'''
单张脸，Dlib 的 API
'''
win = dlib.image_window()
#img = dlib.load_rgb_image('I2G\man.jpeg')
img = dlib.load_rgb_image(r'I2G\faces.jpg')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('I2G\shape_predictor_68_face_landmarks.dat')
dets = detector(img, 1)

win.clear_overlay()
win.set_image(img)

for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        shape = predictor(img,d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  shape.part(1)))
        win.add_overlay(shape)

win.add_overlay(dets)
dlib.hit_enter_to_continue()