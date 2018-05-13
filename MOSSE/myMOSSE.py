#!/usr/bin/env python3


import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

class MOSSE:
    def __init__(self, frame, rect):
        org = frame.copy()
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2 - h)//2
        self.pos = x,y = x1+0.5*(w-1), y1 + 0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w,h), (x,y))

        self.win = cv2.createHanningWindow((w,h), cv2.CV_32F)
        #expected response
        g = np.zeros((h,w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1,-1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags = cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        
        for i in range(128):
            f = self.preprocess(self.rnd_warp(img), self.win)
            F = cv2.dft(f, flags = cv2.DFT_COMPLEX_OUTPUT)

            self.H1 += cv2.mulSpectrums(self.G, F, 0, conjB = True)
            self.H2 += cv2.mulSpectrums(     F, F, 0, conjB = True)
           
        self.update_kernel()    #update kernel
        self.update(frame)  #


    def update(self, frame, rate = 0.125):
        (x,y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x,y))
        img = self.preprocess(img, self.win)
        self.last_resp, (dx, dy), self.psr = self.correlation(img)
        self.good = self.psr > 7.0 
        if not self.good:   
            return 
        
        self.pos = x + dx, y + dy 
        self.last_img = img = cv2.getRectSubPix(frame, (w,h), self.pos)
        img = self.preprocess(img)

        F = cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, F, 0, conjB = True)
        H2 = cv2.mulSpectrums(     F, F, 0, conjB = True)
        self.H1 = rate * H1 + (1.0-rate) * self.H1
        self.H2 = rate * H2 + (1.0-rate) * self.H2
        self.update_kernel()

    def correlation(self, img):
        G = cv2.mulSpectrums(cv2.dft(img, flags = cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB = True)
        resp = cv2.idft(G, flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape

        _, maxVal, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (maxVal - smean) / (sstd + 1e-5)

        return resp, (mx - w//2, my -h//2), psr


    #H = H1/H2
    def update_kernel(self):
        self.H = self.divSpec(self.H1, self.H2)
        self.H[..., -1] *= -1
   

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
        x1, y1, x2, y2 = int(x - 0.5*w) , int(y - 0.5*h), int(x + 0.5*w), int(y + 0.5*h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 1)

        if self.good:
            cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
            print('good--PSR:', self.psr)
        else:
            cv2.line(img, (x1, y1), (x2,y2), (0,0,255))
            cv2.line(img, (x2, y1), (x1,y2), (0,0,255))
            print('not found--PSR:', self.psr)
        cv2.putText(img,'PSR:%.2f'%self.psr,(x1, y2+16),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255), 2)            

    def test(self, frame):
        (x,y), (w,h) = self.pos, self.size
        win = cv2.createHanningWindow((w,h), cv2.CV_32F)

        img1 = self.preprocess(self.last_img, win)
        F1 = cv2.dft(img1, flags = cv2.DFT_COMPLEX_OUTPUT)
        G1 = cv2.mulSpectrums(F1, self.H, 0, conjB = True)
        resp1 = cv2.idft(G1, flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
            
        img = cv2.getRectSubPix(frame, (w, h), (x,y))
        img2 = self.preprocess(img, win)
        F2 = cv2.dft(img2, flags = cv2.DFT_COMPLEX_OUTPUT)
        G2 = cv2.mulSpectrums(F2, self.H, 0, conjB = True)
        resp2 = cv2.idft(G2, flags = cv2.DFT_SCALE|cv2.DFT_REAL_OUTPUT)
    
        merge1 = np.hstack((img1 , resp1))
        merge2 = np.hstack((img2, resp2))
        merge = np.vstack((merge1, merge2))
    

        name = 'comparison--last vs cur.'
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 600,400)
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
    

    r = cv2.selectROI(img, fromCenter = False, showCrosshair = False)
    region = (r[0], r[1], r[0] + r[2], r[1] + r[3])
    print('select roi:', region)
    tracker = MOSSE(img, region)
    tracker.draw_bbox(img)

    print('start tracking...')
    paused = False
    for i in range(1, len(imgsList)):
        frame = cv2.imread(imgsList[i])
        frame = cv2.resize(frame, (256,256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tracker.update(gray, rate = 0.2)

        tracker.draw_bbox(frame)
        tracker.test(gray)

        cv2.imshow('frame', frame)
        ch = cv2.waitKey(30)

        if ch == 27:
            break
        if ch == ord(' '):
            paused = not paused
        if paused:
            cv2.waitKey(0)

