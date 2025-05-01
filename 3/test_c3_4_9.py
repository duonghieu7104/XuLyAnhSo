import tkinter as tk
from tkinter.filedialog import Open
from tkinter.filedialog import asksaveasfilename

import cv2
import numpy as np


import chapter3 as c3
import chapter4 as c4
import chapter9 as c9

class App(tk.Tk):
    def	__init__(self):
        super().__init__()
        self.title('Xử lý ảnh số')
        self.geometry('300x320')
        self.resizable(False, False)
        self.imgin = None
        self.imgout = None
        self.filename = None
       

        menu = tk.Menu(self)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Open Image", command = self.mnu_open_image_click)
        file_menu.add_command(label="Open Color Image", command = self.mnu_open_color_image_click)

        file_menu.add_command(label="Save Image", command = self.mnu_save_image_click)
     


        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        

        chapter3_menu = tk.Menu(menu, tearoff=0)
        chapter3_menu.add_command(label="Negative", command = self.mnu_c3_negative_click)
        chapter3_menu.add_command(label="NegativeColor", command = self.mnu_c3_negative_color_click)
        chapter3_menu.add_command(label="Logarit", command = self.mnu_c3_logarit_click)
        chapter3_menu.add_command(label="Power", command = self.mnu_c3_power_click)
        chapter3_menu.add_command(label="Piecewise Line", command = self.mnu_c3_piecewise_line__click)
        chapter3_menu.add_command(label="Histogram", command = self.mnu_c3_histogram__click)
        chapter3_menu.add_command(label="Hist Equal", command = self.mnu_c3_histequal__click)
        chapter3_menu.add_command(label="Hist Equal Color", command = self.mnu_c3_histequal_color_click)
        chapter3_menu.add_command(label="Local Hist", command = self.mnu_c3_histequal_local_click)
        chapter3_menu.add_command(label="Hist Stat", command = self.mnu_c3_histequal_stat_click)
        chapter3_menu.add_command(label="Smooth Box", command = self.mnu_c3_smooth_box_click)
        chapter3_menu.add_command(label="Smooth Gauss", command = self.mnu_c3_smooth_gauss_click)
        chapter3_menu.add_command(label="Median Filter", command = self.mnu_c3_smooth_filter_click)
        chapter3_menu.add_command(label="Sharpening", command = self.mnu_c3_sharpening_click)
        chapter3_menu.add_command(label="Sharpening Mask", command = self.mnu_c3_sharpening_mask_click)
        chapter3_menu.add_command(label="Grandient", command = self.mnu_c3_grandient_click)
        menu.add_cascade(label="Chapter3", menu=chapter3_menu)

        chapter4_menu = tk.Menu(menu, tearoff=0)
        chapter4_menu.add_command(label="Spectrum", command = self.mnu_c4_spectrum_click)
        chapter4_menu.add_command(label="Remove Morie", command = self.mnu_c4_remove_morie_simple_click)
        chapter4_menu.add_command(label="Remove Inference", command = self.mnu_c4_remove_inference_click)
        chapter4_menu.add_command(label="Create Motion", command = self.mnu_c4_create_motion_click)
        
        menu.add_cascade(label="Chapter4", menu=chapter4_menu)


        chapter9_menu = tk.Menu(menu, tearoff=0)
        chapter9_menu.add_command(label="Erosion", command = self.mnu_c9_erosion_click)
        chapter9_menu.add_command(label="Dilation", command = self.mnu_c9_dilation_click)
        chapter9_menu.add_command(label="Boundary", command = self.mnu_c9_boundary_click)
        chapter9_menu.add_command(label="Contour", command = self.mnu_c9_contour_click)
        menu.add_cascade(label="Chapter9", menu=chapter9_menu)
 

        self.config(menu=menu)

    def mnu_open_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png *.webp')]
        dlg = Open(self, filetypes = ftypes, title = 'Open Image')
        self.filename = dlg.show()

        if self.filename != '':
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            cv2.imshow('ImageIn', self.imgin)            

    def mnu_open_color_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png')]
        dlg = Open(self, filetypes = ftypes, title = 'Open Image')
        self.filename = dlg.show()

        if self.filename != '':
            self.imgin = cv2.imread(self.filename, cv2.IMREAD_COLOR)
            cv2.imshow('ImageIn', self.imgin)            

    def mnu_save_image_click(self):
        ftypes = [('Images', '*.jpg *.tif *.bmp *.png')]
        filenameout = asksaveasfilename(title = 'Image Save', filetypes = ftypes, 
                                initialfile = self.filename) 
        if filenameout is not None:
            cv2.imwrite(filenameout, self.imgout)

    
    def mnu_c3_negative_click(self):
        self.imgout = c3.Negative(self.imgin)  
        cv2.imshow('ImageOut', self.imgout)      
    
    def mnu_c3_negative_color_click(self):
        self.imgout = c3.NegativeColor(self.imgin)  
        cv2.imshow('ImageOut', self.imgout)   
    
    def mnu_c3_logarit_click(self):
        self.imgout = c3.Logarit(self.imgin)    
        cv2.imshow('ImageOut', self.imgout)   

    def mnu_c3_power_click(self):
        self.imgout = c3.Power(self.imgin)  
        cv2.imshow('ImageOut', self.imgout) 

    def mnu_c3_piecewise_line__click(self):
        self.imgout = c3.PiecewiseLine(self.imgin)  
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_histogram__click(self):
        self.imgout = c3.Histogram(self.imgin)  
        cv2.imshow('ImageOut', self.imgout) 

    def mnu_c3_histequal__click(self):
        #self.imgout = c3.HistEqual(self.imgin)
        self.imgout = cv2.equalizeHist(self.imgin)
        cv2.imshow('ImageOut', self.imgout) 

    def mnu_c3_histequal_color_click(self):
        self.imgout = c3.HistEqualColor(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_histequal_local_click(self):
        self.imgout = c3.LocalHist(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_histequal_stat_click(self):
        self.imgout = c3.HistStat(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_smooth_box_click(self):
        self.imgout = cv2.boxFilter(self.imgin, cv2.CV_8UC1, (21, 21))
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_smooth_gauss_click(self):
        self.imgout = cv2.GaussianBlur(self.imgin, (43, 43), 7.0)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_smooth_filter_click(self):
        self.imgout = cv2.medianBlur(self.imgin, 5)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_sharpening_click(self):
        self.imgout = c3.Sharpening(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_sharpening_mask_click(self):
        self.imgout = c3.SharpeningMask(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c3_grandient_click(self):
        self.imgout = c3.SharpeningMask(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c4_spectrum_click(self):
        self.imgout = c4.Spectrum(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c4_remove_morie_simple_click(self):
        self.imgout = c4.RemoveMorie(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c4_remove_inference_click(self):
        self.imgout = c4.RemoveInference(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c4_create_motion_click(self):
        self.imgout = c4.CreateMotion(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

        
    

    def mnu_c9_erosion_click(self):
        self.imgout = c9.Erosion(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c9_dilation_click(self):
        self.imgout = c9.Dilation(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c9_boundary_click(self):
        self.imgout = c9.Boundary(self.imgin)
        cv2.imshow('ImageOut', self.imgout)

    def mnu_c9_contour_click(self):
        self.imgout = c9.Contour(self.imgin)
        cv2.imshow('ImageOut', self.imgout)
if __name__	==	"__main__":
    app	=	App()
    app.mainloop()