import cv2
import os
import _thread as thread
import tqdm

def video_processing(threadid):
    video_dir = r'./dataset/videos'
    video_list = os.listdir(video_dir)
    filenum = len(video_list)
    # 创建一个级联分类器 加载一个.xml分类器文件 它既可以是Haar特征也可以是LBP特征的分类器
    face_detect = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    for i in range (1000):
        if i >= 1000:
            break
        video = video_list[i]
        if "mp4" in video:
            print("starting video",video)
            # 读取视频片段
            video_path = video_dir + '\\{}'.format(video)
            cap = cv2.VideoCapture(video_path)
            t=0
            while True:
                ret, frame = cap.read()
                print('a')
                if not ret:  # 读完视频后falg返回False
                    break
                print('b')
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
                # 灰度处理
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 多个尺度空间进行人脸检测   返回检测到的人脸区域坐标信息
                face_zone = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
                dirs = './dataset/original/{}'.format(video)
                if not os.path.exists(dirs):  # 如果不存在路径，则创建这个路径，关键函数就在这两行，其他可以改变
                    os.makedirs(dirs)
                else:
                    if t == 0:
                        break
                # 绘制矩形和圆形检测人
                for x, y, w, h in face_zone:
                    #cv2.rectangle(frame, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
                    #cv2.circle(frame, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
                    #cv2.imshow('video', frame[y-20:y+h+20,x-10:x+w+10])
                    img = frame[y-30:y+h+20,x-10:x+w+10]
                    try:
                        path = dirs + '\\image{}.jpg'.format(t)
                        if not os.path.exists(path):
                            print(path)
                            cv2.imwrite(path, img)
                            print("successfully write image")
                        else:
                            break
                    except Exception:
                        print("unable to write image")
                # 显示图片
                #cv2.imshow('video', frame)
                # 设置退出键和展示频率
                #if ord('q') == cv2.waitKey(40):
                #    break
                t+=1
            cap.release()
        else:
            print(f'{i} passed')
        print(f'{i} finished')
    # 释放资源
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    video_processing(0)
