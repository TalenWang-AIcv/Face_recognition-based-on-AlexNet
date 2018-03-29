# -*- use coding:utf-8 -*-

import cv2
import dlib
import os


class Camera_reader(object):
    # 在初始化camera的时候建立模型，并加载已经训练好的模型
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.name = ''
        self.mkPath = ''
        self.counter = 0
        self.max_cap_img = 200
        self.ret = True
        self.size = 224

        self.dlib_face_detector = dlib.get_frontal_face_detector()
        self.opencv_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

    def mkdir(self, path):
        # 去除首位空格
        path = path.strip()
        # 去除尾部 / 符号
        path = path.rstrip("/")
        # 判断路径是否存在
        # 存在     True
        # 不存在   False
        isExists = os.path.exists(path)
        # 判断结果
        if not isExists:
            # 如果不存在则创建目录
            print path+' 创建成功'
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print path+' 目录已存在'
            return False

    def create_dir(self):
        self.name = raw_input("Please input your name:\n")
        print "Hello,", self.name
        self.mkPath = './Train_data/' + self.name
        self.mkdir(self.mkPath)

    def dlib_pick_face(self, imgname):
        ret, frame = self.camera.read()
        if ret:
            # using dlib to detect faces
            faces = self.dlib_face_detector(frame, 1)
            print("There are %s face(s)!" % len(faces))
            for d in faces:
                # Base on time name the files
                # listStr = [str(int(time.time()))]
                # fileName = ''.join(listStr)
                # resize faces as 224*224, and save!
                x = dlib.rectangle.left(d)
                y = dlib.rectangle.top(d)
                x_w = dlib.rectangle.right(d)
                y_h = dlib.rectangle.bottom(d)
                image = cv2.resize(frame[y:y_h, x:x_w], (self.size, self.size))
                # cv2.imwrite(self.mkPath + os.sep + self.name + '%s.jpg' % fileName, image)
                cv2.imwrite('%s.jpg' % imgname, image)
                # using opencv draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x_w, y_h), (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, 'Rolling...Please keep your face',
                        (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
            cv2.putText(frame, 'inside green rectangle!',
                        (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
            cv2.imshow("dlib", frame)
        else:
            print '[Error]:Camera read false!'

    def opencv_pick_faces(self, imgname):
        self.ret, frame = self.camera.read()
        if self.ret:
            # Convert the captured frame into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Get all face from the video frame using opencv
            faces = self.opencv_face_detector.detectMultiScale(gray, 1.25, 5)
            print("There are %s face(s)!" % len(faces))
            for (x, y, w, h) in faces:
                # Base on time name the files
                # listStr = [str(int(time.time()))]
                # fileName = ''.join(listStr)
                # resize faces as 224*224, and save!
                image = cv2.resize(frame[y:(y + h), x:(x + w)], (self.size, self.size))
                # cv2.imwrite(self.mkPath + os.sep + self.name + '%s.jpg' % fileName, image)
                cv2.imwrite(imgname, image)
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Rolling...Please keep your face',
                        (0, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
            cv2.putText(frame, 'inside green rectangle!',
                        (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 255))
            cv2.imshow('opencv', frame)
        else:
            print '[Error]:Camera read false!'


# if __name__ == "__main__":
#     print("[INFO] Reading system camera...")
#     cam = Camera_reader()
#     cam.create_dir()
#     # start the app
#     while True:
#         cam.dlib_pick_face('wang.jpg')
#         # cam.opencv_pick_faces()
#         if cv2.waitKey(1) & 0xff is 27:
#             break
#
#     cam.camera.release()
#     cv2.destroyAllWindows()
