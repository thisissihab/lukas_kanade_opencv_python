import numpy as np
import cv2

#Create the video capture object
cap = cv2.VideoCapture('sample.mp4')


fp = dict( maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )

lkp = dict( winSize = (15, 15),maxLevel = 3,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10, 0.03))
  
color = np.random.randint(0, 255, (100, 3))

_ , prev_frame = cap.read()
old_gray = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)

#Shi-Tomashi Corner  Detection
kp1 = cv2.goodFeaturesToTrack(old_gray, mask = None,**fp)

mask = np.zeros_like(prev_frame)

while(1):
      
    try:
        _, frame = cap.read()
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # calculate the optical flows
        kp2, st, err = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,kp1, None,**lkp)

        # Find the good points
        good_new = kp2[st == 1]
        good_old = kp1[st == 1]

        # place the trackers on the video
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)),color[i].tolist(), 2)
          
            frame = cv2.circle(frame, (int(a),int(b)), 5,color[i].tolist(), -1)
        img = cv2.add(frame, mask)
        img = cv2.resize(img, (450, 300), interpolation = cv2.INTER_AREA)
  
        cv2.imshow('Video', img)

        # Updating the previous frames and previous points
        old_gray = frame_gray.copy()
        kp1 = good_new.reshape(-1, 1, 2)
    
      
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    except:
        break
  
cv2.destroyAllWindows()
cap.release()
