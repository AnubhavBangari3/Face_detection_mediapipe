import cv2
import mediapipe as mp
import time
#https://google.github.io/mediapipe/solutions/face_detection#python-solution-api
pTime,cTime=0,0
cap=cv2.VideoCapture(0)

mp_face=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceD=mp_face.FaceDetection()


while True:
    success,img=cap.read()
    imgRB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceD.process(imgRB)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            mpDraw.draw_detection(img,detection)
            
            h,w,c=img.shape
            b=detection.location_data.relative_bounding_box
            bbox=int(b.xmin*w),int(b.ymin*h),int(b.width*w),int(b.height*h)
            
            cv2.rectangle(img,bbox,(255,0,0),4)
            cv2.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
            
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,str(int(fps)),(10,78),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    cv2.imshow("Image",img)
    key=cv2.waitKey(1)
    #ascii for q and Q to quit
    if key==81 or key==113:
        break
cap.release()