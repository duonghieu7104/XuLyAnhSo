import numpy as np
import cv2
L = 256

def Spectrum(imgin):
    f  = imgin.astype(np.float32)/(L-1)
    #Buoc1: DFT
    F = np.fft.fft2(f)
    #Buoc2: Shift vao the center of the inmage
    F = np.fft.fftshift(F)

    #Buoc5: Tinh Spectrum
    S = np.sqrt(F.real**2 + F.imag**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)

    return imgout

def CreateMoireFilter(M, N, D0=9):
    H = np.ones((M, N), np.complex64)

    # Danh sách tọa độ các điểm cần lọc (và đối xứng chính xác)
    coords = [
        (44, 55), (85, 55), (41, 111), (81, 111),
        (M - 44, N - 55), (M - 85, N - 55), (M - 41, N - 111), (M - 81, N - 111)
    ]

    for u in range(M):
        for v in range(N):
            for (uc, vc) in coords:
                Duv = np.sqrt((u - uc) ** 2 + (v - vc) ** 2)
                if Duv <= D0:
                    H[u, v] = 0.0
                    break  # Không cần kiểm tra các điểm còn lại

    return H



def FrequencyFiltering(imgin, H):
    f  = imgin.astype(np.float32)
    #Buoc1: DFT
    F = np.fft.fft2(f)
    #Buoc2: Shift vao the center of the inmage
    F = np.fft.fftshift(F)
    #Buoc3: Nhan F voi H
    G = F*H

    #Buoc4: Shift return
    G = np.fft.ifftshift(G)

    #Buoc5: IDFT
    g = np.fft.ifft2(G)
    gR = np.clip(g.real, 0, L-1)
    imgout = gR.astype(np.uint8)

    return imgout

def CreateInferenceFilter(M, N, D0=7, D1=7):
    H = np.ones((M, N), np.complex64)

    for u in range(M):
        for v in range(N):
            # Loại bỏ một dải dọc nhỏ quanh v = N//2 (theo chiều ngang)
            if abs(u - M//2) > D0 and abs(v - N//2) <= D1:
                H[u, v] = 0.0  # lọc bỏ nhiễu theo trục dọc

    return H


def CreateMotionFilter(M, N):
    H = np.zeros((M, N), np.complex64)
    T = 1.0
    a = 0.1
    b = 0.1
    phi_prev = 0.0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if abs(phi) < 1.0e-6:
                phi = phi_prev
                
            else:
                RE = T*np.sin(phi)*np.cos(phi)/phi
                IM = -T*np.sin(phi)*np.sin(phi)/phi
            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    
    return H




def RemoveMorie(imgin):
    M, N = imgin.shape
    H = CreateMoireFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout

def RemoveInference(imgin):
    M, N = imgin.shape
    H = CreateInferenceFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


def CreateMotion(imgin):
    M, N = imgin.shape
    H = CreateMotionFilter(M, N)
    imgout = FrequencyFiltering(imgin, H)
    return imgout


