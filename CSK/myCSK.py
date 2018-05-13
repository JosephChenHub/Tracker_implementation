#!/usr/bin/env python3


import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


class CSK:
    def __init__(self, frame, rect): #given a initial frame, and ROI region (top-left,down-right)
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        x1, y1, x2, y2 = rect
        w_, h_ = x2-x1, y2-y1
        self.factor = 2 # region is factor * roi
        x1, y1 = (x1+x2- w_)//2, (y1+y2 - h_ )//2
        self.pos = x,y = x1+0.5*(w_-1), y1 + 0.5*(h_-1) #center 
        w, h = map(cv2.getOptimalDFTSize, [int(self.factor*w_), int(self.factor*h_)]) #get optimal DFT size
        self.size = w, h 

        img = cv2.getRectSubPix(frame, self.size, self.pos) #will auto padding if size exceeds boundaries
        self.win = cv2.createHanningWindow(self.size, cv2.CV_32F)
        self.X = self.preprocess(img, self.win) #first patch
      
        self.Y = self.target(w/self.factor, h/self.factor, self.factor) #expected response, gaussian distri.
 
        self.sigma = 0.2
        self.lamb = 0.01
        self.dgk = self.gaussian_kernel

        self.alphaf = self.training(self.X, self.Y, self.sigma, self.lamb)
    

    def update(self, frame, rate = 0.075):
        
        (x,y), (w, h) = self.pos, self.size
        img = cv2.getRectSubPix(frame, self.size, self.pos)
        img = self.preprocess(img, self.win) #normalization , window
        self.last_img = img
        #detection
        resp = self.detection(self.alphaf, self.X, img, self.sigma)
        _, mVal, _, (mx, my) = cv2.minMaxLoc(resp)
        self.resp = resp
        
        dx = -mx + w/2
        dy = -my + h/2

        self.psr = (mVal - resp.mean())/(resp.std() + 1e-5)
         
        self.good = self.psr > 8.0 
        if not self.good:   
            return 
        
        
        
        #update position
        self.pos = x + dx, y + dy
        #training
        img_next = cv2.getRectSubPix(frame, self.size, self.pos)
        img_next = self.preprocess(img_next, self.win)

        #adaption 
        self.alphaf = rate* self.training(img_next, self.Y, self.sigma, self.lamb) + (1-rate)* self.alphaf
        self.X = (1-rate)*self.X + rate * img_next
      

        return self.pos

    def training(self, x, y, sigma, lamb):
        k = self.dgk(x, x, sigma) #Kxx
        alphaf = self.divSpec( cv2.dft(y, flags = cv2.DFT_COMPLEX_OUTPUT),
                    cv2.dft(k, flags= cv2.DFT_COMPLEX_OUTPUT) + lamb) #alphaf = F(y) \ (F(Kxx) + lambda)
        
        return np.float32(alphaf) 

    def detection(self, alphaf, x, z, sigma):
        k = self.dgk(x, z, sigma) #Kxz
        resp = cv2.idft( cv2.mulSpectrums(alphaf , cv2.dft(k, flags = cv2.DFT_COMPLEX_OUTPUT), 0, conjB = False),
                        flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT) # F(resp) = alphaf * F(Kxz) 
        return resp
        
    
    def target(self, w, h, factor = 2):
        pad_h = factor * h
        pad_w = factor * w
        s = np.sqrt( pad_h * pad_w ) / 16

        j = np.arange(0, pad_w)
        i = np.arange(0, pad_h)
        J, I = np.meshgrid(j, i)
        y = np.exp(-1/s**2 * ( (J - pad_w//2)**2 + (I - pad_h//2)**2))

        return y


    def gaussian_kernel(self,x1, x2, sigma):
        F1 = cv2.dft(x1, flags = cv2.DFT_COMPLEX_OUTPUT)
        F2 = cv2.dft(x2, flags = cv2.DFT_COMPLEX_OUTPUT)
        tmp = (F1[..., 0] + 1j*F1[..., 1]) * (F2[..., 0] - 1j * F2[..., 1]) #F1*F2.conj()
        tmp2 = np.dstack([np.real(tmp), np.imag(tmp)])

        c = np.fft.fftshift(cv2.idft(tmp2, flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)) 
        d = np.dot( np.conj(x1.flatten(1)) , x1.flatten(1)) + np.dot(np.conj(x2.flatten(1)), x2.flatten(1)) - 2 * c
        k = np.exp(-0.5/sigma**2 * np.abs(d) / np.size(x1))

        return k

    def linear_kernel(self, x1, x2, sigma):
        F1 = cv2.dft(x1, flags = cv2.DFT_COMPLEX_OUTPUT)
        F2 = cv2.dft(x2, flags = cv2.DFT_COMPLEX_OUTPUT)
        
        tmp = (F1[..., 0] + 1j*F1[..., 1]) * (F2[..., 0] - 1j * F2[..., 1]) #F1*F2.conj()
        tmp2 = np.dstack([np.real(tmp), np.imag(tmp)])
        k = np.fft.fftshift(cv2.idft(tmp2, flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT))

        return k


    #log, normalization, window
    def preprocess(self, img, win = None):
        img = np.log(np.float32(img) + 1.0) 
        img = (img - img.mean()) / (img.std() + 1e-5)

        if win is not None:
            return img * win
        else:
            return img

    #return A/B
    def divSpec(self, A, B):
        Ar, Ai = A[..., 0], A[..., 1]
        Br, Bi = B[..., 0], B[..., 1]
        C = (Ar + 1j*Ai) / (Br + 1j*Bi)
        C = np.dstack([np.real(C), np.imag(C)]).copy()

        return C
 
    
    def rnd_warp(self, img):
        h, w = img.shape[:2]
        T = np.zeros((2,3))
        coef = 0.2
        ang = (np.random.rand() - 0.5)*coef
        c, s = np.cos(ang), np.sin(ang)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:2, :2] += (np.random.rand(2,2) - 0.5) * coef
        c = (w/2, h/2)
        T[:, 2] = c - np.dot(T[:2, :2] , c)

        return cv2.warpAffine(img, T, (w,h), borderMode = cv2.BORDER_REFLECT)

    def draw_bbox(self, img):
        (x,y) ,(w, h) = self.pos, self.size
        w = w/self.factor
        h = h/self.factor

        x1, y1, x2, y2 = int(x - 0.5*w) , int(y - 0.5*h), int(x + 0.5*w), int(y + 0.5*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)

        if self.good:
            cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
            print('good--PSR:{}'.format(self.psr))
        else:
            cv2.line(img, (x1, y1), (x2,y2), (0,0,255))
            cv2.line(img, (x2, y1), (x1,y2), (0,0,255))
            print('not found--PSR:', self.psr)
        cv2.putText(img,'PSR:%.2f'%self.psr,(x1, y2+16),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2)            

    def test(self, frame):
        name = 'resp.'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600, 400)
        h,w = self.last_img.shape
        mid = np.ones((h, 15))
        merge = np.hstack((self.last_img, mid))
        merge = np.hstack((merge, self.resp))

        cv2.imshow(name, merge)

if __name__ == '__main__':
    
    input_path = None
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print('please input the imgs file...')
        exit()

    import os
    imgsList = []
    for root, dir, files in os.walk(input_path):
        for f in files:
            if ('.jpg' in f) or ('.png' in f) or('.bmp' in f):
                tmp = os.path.join(root, f)
                imgsList.append(tmp)

    imgsList = sorted(imgsList)
    #first frame for ROI select
    img = cv2.imread(imgsList[0])
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    

    r = cv2.selectROI(img, fromCenter = False, showCrosshair = False)
    region = (r[0], r[1], r[0] + r[2], r[1] + r[3])
    print('select roi:', region)
    tracker = CSK(gray, region)

    print('start tracking...')
    paused = False
    for i in range(1, len(imgsList)):
        frame = cv2.imread(imgsList[i])
        frame = cv2.resize(frame, (256,256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracker.update(gray)

        tracker.draw_bbox(frame)
        tracker.test(frame)
    
        cv2.imshow('frame', frame)
        ch = cv2.waitKey(30)

        if ch == 27:
            break
        if ch == ord(' '):
            paused = not paused
        if paused:
            cv2.waitKey(0)

