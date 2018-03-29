# -*- use coding:utf-8 -*-
import cv2
import time
import os


def mkdir(path):
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
        print path + ' 创建成功'
        # 创建目录操作函数
        os.makedirs(path)
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print path + ' 目录已存在'
        return False


size = 224
videopath = '/home/talentwong/Documents/wtl/Video - AVI/'
video = videopath + '58106a8e-11b2-40b2-8221-bfabf1e17f3f.avi'
cap = cv2.VideoCapture(video)
opencv_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

path = './Train_data/train053'
if __name__ == '__main__':
    mkdir(path)

    while True:
        listStr = [str(int(time.time()))]
        fileName = ''.join(listStr)
        imgname = 'train' + '%s.jpg' % fileName
        # get a frame
        ret, frame = cap.read()
        if ret:
            # Convert the captured frame into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Get all face from the video frame using opencv
            faces = opencv_face_detector.detectMultiScale(gray, 1.25, 5)
            for (x, y, w, h) in faces:
                # Base on time name the files
                # listStr = [str(int(time.time()))]
                # fileName = ''.join(listStr)
                # resize faces as 224*224, and save!
                image = cv2.resize(frame[y:(y + h), x:(x + w)], (size, size))
                # cv2.imwrite(self.mkPath + os.sep + self.name + '%s.jpg' % fileName, image)
                cv2.imwrite(path + os.sep + imgname, image)
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # show a frame
            cv2.imshow("capture", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

