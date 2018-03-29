# -*- coding:utf-8 -*-
from Cap_camera import Camera_reader
import Tkinter as tk
import time
import cv2
import os


class MainWindow(object):
    def __init__(self):
        self.mkpath = ''
        self.name = ''

        self.mainwindow = tk.Tk()
        self.mainwindow.title('Main window')
        self.mainwindow.geometry('300x300')

        global enrollentry, name_entry, keys_entry
        global enrollwindow, sign_in_window

    # -------------------------主界面Enroll按键---------------------------------
    def Enroll_Button(self):
        enroll_B = tk.Button(self.mainwindow, text='Enroll', font=('Arial', 10),
                             width=15, height=2, command=self.enroll)
        enroll_B.place(x=150, y=50, anchor=tk.CENTER)
        # roll.pack()

    # -------------------------Enroll界面---------------------------------
    def enroll(self):
        global enrollentry, enrollwindow
        enrollwindow = tk.Tk()
        enrollwindow.title('Enroll')
        enrollwindow.geometry('300x230')

        note = tk.Label(enrollwindow,
                        text='Please input your name \nclick Save to enroll ^_^',
                        font=('Arial', 10),
                        width=30, height=2)
        note.place(x=0, y=10, anchor=tk.NW)

        user = tk.Label(enrollwindow,
                        text='User name:',  # 使用 textvariable 替换 text, 因为这个可以变化
                        font=('Arial', 10),
                        width=12, height=2)
        user.place(x=50, y=60, anchor=tk.NW)

        enrollentry = tk.Entry(enrollwindow, show=None)
        enrollentry.place(x=65, y=100, anchor=tk.NW)

        save_B = tk.Button(enrollwindow, text='Save',
                           font=('Arial', 10), width=10,
                           height=1, command=self.makedir)
        save_B.place(x=220, y=150, anchor=tk.CENTER)

        Quit_B = tk.Button(enrollwindow, text='Quit',
                           font=('Arial', 10), width=10,
                           height=1, command=self.Quit)
        Quit_B.place(x=80, y=150, anchor=tk.CENTER)

    # -------------------------Enroll界面注册生成相应文件夹---------------------------------
    def makedir(self):
        global enrollentry
        self.name = enrollentry.get()
        self.mkpath = './Train_data/' + self.name
        self.mkDir(self.mkpath)

        cam = Camera_reader()
        # start the app
        while True:
            listStr = [str(int(time.time()))]
            fileName = ''.join(listStr)
            imgname = self.mkpath + os.sep + self.name + '%s.jpg' % fileName
            # cam.dlib_pick_face(imgname)
            cam.opencv_pick_faces(imgname)
            # cam.opencv_pick_faces()
            if cv2.waitKey(1) & 0xff is 27:
                break

        cam.camera.release()
        cv2.destroyAllWindows()

    def mkDir(self, path):
        global enrollwindow
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
            label_text = str('[INFO]' + path + '\nhas been built successfully!')
            print label_text
            massage = tk.Label(enrollwindow,
                               bg='white',
                               text=label_text,  # 使用 textvariable 替换 text, 因为这个可以变化
                               font=('Arial', 10),
                               width=30, height=2)
            massage.place(x=0, y=180, anchor=tk.NW)
            # 创建目录操作函数
            os.makedirs(path)
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            label_text = str('[INFO]' + path + '\nhas already existed!')
            print label_text
            massage = tk.Label(enrollwindow,
                               bg='yellow',
                               text=label_text,  # 使用 textvariable 替换 text, 因为这个可以变化
                               font=('Arial', 10),
                               width=30, height=2)
            massage.place(x=0, y=180, anchor=tk.NW)
            return False

    def Quit(self):
        global enrollwindow
        enrollwindow.destroy()

    # -------------------------主界面Manager按键---------------------------------
    def Manager_Button(self):
        manage_B = tk.Button(self.mainwindow, text='Manage', font=('Arial', 10),
                             width=15, height=2, command=self.sign_in)
        manage_B.place(x=150, y=110, anchor=tk.CENTER)

    # -------------------------Manager登录界面---------------------------------
    def sign_in(self):
        global name_entry, keys_entry
        global sign_in_window
        sign_in_window = tk.Tk()
        sign_in_window.title('Sign in')
        sign_in_window.geometry('300x230')

        note = tk.Label(sign_in_window,
                        text='Please input your name \nclick Sign_in to manage ^.^',
                        font=('Arial', 10),
                        width=30, height=2)
        note.place(x=0, y=10, anchor=tk.NW)

        manager_L = tk.Label(sign_in_window,
                             text='Manager name:',  # 使用 textvariable 替换 text, 因为这个可以变化
                             font=('Arial', 10),
                             width=12, height=2)
        manager_L.place(x=15, y=60, anchor=tk.NW)
        name_entry = tk.Entry(sign_in_window, show=None)
        name_entry.place(x=15, y=100, anchor=tk.NW)

        keys = tk.Label(sign_in_window,
                        text='Keys:',  # 使用 textvariable 替换 text, 因为这个可以变化
                        font=('Arial', 10),
                        width=8, height=2)
        keys.place(x=0, y=120, anchor=tk.NW)
        keys_entry = tk.Entry(sign_in_window, show='*')
        keys_entry.place(x=15, y=160, anchor=tk.NW)

        manage_B = tk.Button(sign_in_window, text='Sign_in', font=('Arial', 10),
                             width=6, height=3, command=self.Sign_in)
        manage_B.place(x=240, y=140, anchor=tk.CENTER)

    # -------------------------Manager登录提示---------------------------------
    def Sign_in(self):
        global name_entry, keys_entry
        global sign_in_window
        manager = name_entry.get()
        keys = keys_entry.get()
        if manager != 'Talent':
            message = tk.Label(sign_in_window,
                               bg='red',
                               text='[ERROR]:' + manager + ' is not a manager!',
                               font=('Arial', 10),
                               width=30, height=2)
            message.place(x=0, y=185, anchor=tk.NW)
        else:
            if keys != '123456':
                key_mess = tk.Label(sign_in_window,
                                    bg='red',
                                    text='[ERROR]:Wrong keys!',
                                    font=('Arial', 10),
                                    width=30, height=2)
                key_mess.place(x=0, y=185, anchor=tk.NW)
            else:
                message = tk.Label(sign_in_window,
                                   bg='green',
                                   text='Welcome! ' + manager,  # 使用 textvariable 替换 text, 因为这个可以变化
                                   font=('Arial', 10),
                                   width=30, height=2)
                message.place(x=0, y=185, anchor=tk.NW)

                self.manager_operation()

    # -------------------------manager管理界面---------------------------------
    def manager_operation(self):
        pass


if __name__ == '__main__':
    test = MainWindow()
    test.Enroll_Button()
    test.Manager_Button()
    test.mainwindow.mainloop()
