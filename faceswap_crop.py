import cv2
import os
import _thread as thread
from tqdm import tqdm

video_dir = r"./dataset/manipulated/FaceSwap/c23/videos"
video_list = os.listdir(video_dir)
filenum = len(video_list)
# 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
face_detect = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
for i in tqdm(range(1000)):
    video = video_list[i]
    idx = video.split('_')[0]
    if idx < 700:
        pass
    else:
        if "mp4" in video:
            print("starting video", video)
            # 读取视频片段
            video_path = os.path.join(video_dir, video)
            cap = cv2.VideoCapture(video_path)
            t = 0
            while True:
                ret, frame = cap.read()
                if not ret:  # 读完视频后flag返回False
                    print(f"{video} finished")
                    break
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                # 灰度处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
                face_zone = face_detect.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=8
                )
                dirs = os.path.join("./dataset/testset/faceswap", video)
                if not os.path.exists(dirs):
                    os.makedirs(dirs)
                else:
                    if t == 0:
                        break
                for x, y, w, h in face_zone:
                    img = frame[y - 30 : y + h + 20, x - 10 : x + w + 10]
                    try:
                        path = os.path.join(dirs, f"image{t}.jpg")
                        if not os.path.exists(path):
                            print(path)
                            cv2.imwrite(path, img)
                            # print("successfully write image")
                        else:
                            break
                    except Exception:
                        print("unable to write image")
                t += 1
            cap.release()
