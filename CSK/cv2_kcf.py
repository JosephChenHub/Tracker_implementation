import cv2
import sys

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
    tracker = cv2.TrackerKCF_create()
#    tracker = cv2.TrackerTLD_create()
    tracker.init(img, r)
    
    print('start tracking...')
    paused = False
    for i in range(1, len(imgsList)):
        frame = cv2.imread(imgsList[i])
        frame = cv2.resize(frame, (256,256))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ok, bbox = tracker.update(frame)
    
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 1)
        else:
            cv2.putText(frame, "Tracking failure", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 1)

        cv2.imshow('frame', frame)
        ch = cv2.waitKey(30)

        if ch == 27:
            break
        if ch == ord(' '):
            paused = not paused
        if paused:
            cv2.waitKey(0)

