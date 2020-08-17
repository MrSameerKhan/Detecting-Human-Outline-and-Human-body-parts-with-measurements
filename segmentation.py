import cv2
import time
import numpy as np

MODE = "COCO"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]




frame1 = cv2.imread("imgs/single1.jpg")
#cv2.imshow("orignal",frame1)
frame=cv2.resize(frame1, (640, 960)) 
#cv2.imshow("resized",frame2)
#cv2.waitKey(0)
#frame=cv2.imread(frame2)

frameCopy = np.copy(frame)
frameWidth = frame.shape[1]
frameHeight = frame.shape[0]
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

t = time.time()
# input image dimensions for the network
inWidth = 368
inHeight = 368
inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                          (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()
print("time taken by network : {:.3f}".format(time.time() - t))

H = output.shape[2]
W = output.shape[3]

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Empty list to store the detected keypoints
points = []

for i in range(nPoints):
    # confidence map of corresponding body's part.
    probMap = output[0, i, :, :]

    # Find global maxima of the probMap.
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    
    # Scale the point to fit on the original image
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H

    if prob > threshold : 
        cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

        # Add the point to the list if the probability is greater than the threshold
        points.append((int(x), int(y)))
    else :
        points.append(None)



# Draw Skeleton
for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    #Right Hand
    
    if points[2] and points[3]:
        half=points[2]
        half1=points[3]
        x1=half[0]
        x2=half1[0]
        y1=half[1]
        y2=half1[1]
        mid1x=int((x1+x2)/2)
        mid1y=int((y1+y2)/2)
        #print("----------midx------------")
        #print(int(mid1x))
        #print("----------------midy------------")
        #print(int(mid1y))
       
        #cv2.line(frame, points[2], points[3], (255,0, 0), 2)
        cv2.circle(frame, points[2], 8, (0, 0, 0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[3], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, half, 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        #cv2.ellipse(frame,(mid1x,mid1y),(100,50),0,0,360,255,3)80,50
        cv2.ellipse(frame,(mid1x,mid1y),(30,50),30,0.0,360.0,(255,0,0),4)
    if points[2] and points[5]:
        (mX, mY) = midpoint(points[2], points[4])
        cv2.putText(frame, "Right Hand", (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    if points[3] and points[4]:
        right1=points[3]
        right2=points[4]
        right_x=right1[0]
        right_x1=right2[0]
        right_y=right1[1]
        right_y1=right2[1]
        mid_x=int((right_x+right_x1)/2)
        mid_y=int((right_y+right_y1)/2)
        #print("hahahahhahahahhahahahahahhahahahahah")
        #print(mid_x,mid_y)
        #cv2.line(frame, points[3], points[4], (255,0, 0), 2)
        cv2.circle(frame, points[3], 8, (0, 0, 0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[4], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_x,mid_y),(30,50),30,0.0,360.0,(255,0,0),4)
    #cv2.putText(frame,'right side ',(), font, 4,(0,255,0),2,cv2.LINE_AA)
    
    #Left Hand
    if points[5] and points[6]:
        left=points[5]
        left1=points[6]
        left_x1=left[0]
        left_x2=left1[0]
        left_y1=left[1]
        left_y2=left1[1]
        mid_xx=int((left_x1+left_x2)/2)
        mid_yy=int((left_y1+left_y2)/2)
        #cv2.line(frame, points[5], points[6], (255,0, 0), 2)
        cv2.circle(frame, points[5], 8, (0, 0, 0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[6], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xx,mid_yy),(30,50),-30,0.0,360.0,(255,0,0),4)

    if points[5] and points[7]:
        (mX, mY) = midpoint(points[5], points[7])
        cv2.putText(frame, "Left Hand", (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    if points[6] and points[7]:
        leftt=points[6]
        left1t=points[7]
        left_x1t=leftt[0]
        left_x2t=left1t[0]
        left_y1t=leftt[1]
        left_y2t=left1t[1]
        mid_xxt=int((left_x1t+left_x2t)/2)
        mid_yyt=int((left_y1t+left_y2t)/2)
        #cv2.line(frame, points[6], points[7], (255,0, 0), 2)
        cv2.circle(frame, points[6], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[7], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xxt,mid_yyt),(30,50),-30,0.0,360.0,(255,0,0),4)
    #Right Leg
    if points[8] and points[9]:
        lefttf=points[8]
        left1tf=points[9]
        left_x1tf=lefttf[0]
        left_x2tf=left1tf[0]
        left_y1tf=lefttf[1]
        left_y2tf=left1tf[1]
        mid_xxtf=int((left_x1tf+left_x2tf)/2)
        mid_yytf=int((left_y1tf+left_y2tf)/2)
       # cv2.line(frame, points[8], points[9], (0,255, 0), 2)
        cv2.circle(frame, points[8], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[9], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xxtf,mid_yytf),(30,50),0.0,0.0,360.0,(0,255,0),4)

    if points[8] and points[10]:
        (mX, mY) = midpoint(points[8], points[10])
        cv2.putText(frame, "Right Leg", (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    if points[9] and points[10]:
        lefttm=points[9]
        left1tm=points[10]
        left_x1tm=lefttm[0]
        left_x2tm=left1tm[0]
        left_y1tm=lefttm[1]
        left_y2tm=left1tm[1]
        mid_xxtm=int((left_x1tm+left_x2tm)/2)
        mid_yytm=int((left_y1tm+left_y2tm)/2)
       # cv2.line(frame, points[9], points[10], (0,255, 0), 2)
        cv2.circle(frame, points[9], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[10], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xxtm,mid_yytm),(30,50),0.0,0.0,360.0,(0,255,0),4)
        
    #Left leg
    if points[11] and points[12]:
        leftts=points[11]
        left1ts=points[12]
        left_x1ts=leftts[0]
        left_x2ts=left1ts[0]
        left_y1ts=leftts[1]
        left_y2ts=left1ts[1]
        mid_xxts=int((left_x1ts+left_x2ts)/2)
        mid_yyts=int((left_y1ts+left_y2ts)/2)

        #cv2.line(frame, points[11], points[12], (0,255, 0), 2)
        cv2.circle(frame, points[11], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[12], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xxts,mid_yyts),(30,50),0.0,0.0,360.0,(0,255,0),4)
    if points[11] and points[13]:
        (mX, mY) = midpoint(points[11], points[13])
        cv2.putText(frame, "Left Leg", (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
    if points[12] and points[13]:
        lefttz=points[12]
        left1tz=points[13]
        left_x1tz=lefttz[0]
        left_x2tz=left1tz[0]
        left_y1tz=lefttz[1]
        left_y2tz=left1tz[1]
        mid_xxtz=int((left_x1tz+left_x2tz)/2)
        mid_yytz=int((left_y1tz+left_y2tz)/2)
        
        #cv2.line(frame, points[12], points[13], (0,255, 0), 2)
        cv2.circle(frame, points[12], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[13], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
        cv2.ellipse(frame,(mid_xxtz,mid_yytz),(30,50),0.0,0.0,360.0,(0,255,0),4)
        
    #torso 
    if points[2] and points[5]:
        cv2.line(frame, points[2], points[5], (0,0,255), 2)
        cv2.circle(frame, points[2], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[5], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
    if points[2] and points[8]:
        cv2.line(frame, points[2], points[8], (0,0,255), 2)
        cv2.circle(frame, points[2], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[8], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
    if points[8] and points[11]:
        cv2.line(frame, points[8], points[11], (0,0,255), 2)
        cv2.circle(frame, points[8], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[11], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
    if points[5] and points[11]:
        cv2.line(frame, points[5], points[11], (0,0,255), 2)
        cv2.circle(frame, points[5], 8, (0, 0,0), thickness=-3, lineType=cv2.FILLED)
        cv2.circle(frame, points[11], 8, (0, 0,0), thickness=-1, lineType=cv2.FILLED)
    if points[2] and points[11]:
        (mX, mY) = midpoint(points[2], points[11])
        cv2.putText(frame, "Torso", (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)


        
           
        
          

#    if points[partA] and points[partB]:
#       cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
#       cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

#cv2.boxPoints()
#v2.imshow('Output-Keypoints', frameCopy)
cv2.imshow('Body_Segmentation', frame)


#cv2.imwrite('Output-Keypoints.jpg', frameCopy)
cv2.imwrite('Body_Segmentation.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
cv2.destroyAllWindows()

