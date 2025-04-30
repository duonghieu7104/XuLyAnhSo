import tkinter as tk
from tkinter.filedialog import Open
from tkinter.filedialog import asksaveasfilename

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Xử lý ảnh')
        self.geometry('250x320')
        self.resizable(False, False)
        self.imgin = None
        self.imgout = None
        self.filename = None
        # Thay đổi đường dẫn đến mô hình best.onnx
        self.model = YOLO(r"C:\Users\LENOVO\Downloads\best.onnx", task='detect')

        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.mnu_open_image_click)
        file_menu.add_command(label="Open Color Image", command=self.mnu_open_color_image_click)

        file_menu.add_command(label="Save Image", command=self.mnu_save_image_click)
        file_menu.add_command(label="Predict", command=self.mnu_predict_click)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)
    
        self.config(menu=menu)

    def mnu_open_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png')]
        dlg = Open(self, filetypes=ftypes, title='Open Image')
        self.filename = dlg.show()

        if self.filename != '':
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('ImageIn', self.imgin)            

    def mnu_open_color_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png')]
        dlg = Open(self, filetypes=ftypes, title='Open Image')
        self.filename = dlg.show()

        if self.filename != '':
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_COLOR)
            cv2.imshow('ImageIn', self.imgin)            

    def mnu_save_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png')]
        filenameout = asksaveasfilename(title='Image Save', filetypes=ftypes, 
                                        initialfile=self.filename) 
        if filenameout is not None:
            cv2.imwrite(filenameout, self.imgout)

    def mnu_predict_click(self):
        names = self.model.names
        self.imgout = self.imgin.copy()
        annotator = Annotator(self.imgout)
        results = self.model.predict(self.imgin, conf=0.5, verbose=False)

        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.tolist()
        for box, cls, conf in zip(boxes, clss, confs):
            annotator.box_label(box, label=names[int(cls)] + ' %4.2f' % conf, txt_color=(255, 0, 0), color=(255, 255, 255))
        cv2.imshow('ImageOut', self.imgout)            

if __name__ == "__main__":
    app = App()
    app.mainloop()
