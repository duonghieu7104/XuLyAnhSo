import os
import cv2
import numpy as np

# Đường dẫn nguồn và đích được chỉnh sửa
path = 'C:/Users/LENOVO/Downloads/Nam3.2/XU_LY_ANH_SO/TraiCayScratch/Kiwi/'
path_dest = 'C:/Users/LENOVO/Downloads/Nam3.2/XU_LY_ANH_SO/TraiCay640x640/Kiwi/'

lst_dir = os.listdir(path)
dem = 0
for filename in lst_dir:
    print(filename)
    fullname = path + filename
    imgin = cv2.imread(fullname, cv2.IMREAD_COLOR)
    # Kiểm tra nếu ảnh không hợp lệ
    if imgin is None:
        print(f"Lỗi: Không thể đọc file {fullname}")
        continue
    
    # M: width, N: height, C: channel: 3
    M, N, C = imgin.shape
    if M < N:
        imgout = np.zeros((N, N, C), np.uint8) + 125
        imgout[:M, :N, :] = imgin
    elif M > N:
        imgout = np.zeros((M, M, C), np.uint8) + 125
        imgout[:M, :N, :] = imgin
    else:
        imgout = imgin.copy()
    
    # Resize ảnh về kích thước 640x640
    imgout = cv2.resize(imgout, (640, 640))
    fullname_dest = path_dest + "Kiwi%03d.jpg" % dem
    dem = dem + 1
    cv2.imwrite(fullname_dest, imgout)
