import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from math import sqrt

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

height = 5.10

dpi =96

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

centi=0.032808

x2=1
x1=1
y2=1
y1 = 1280



#dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

print("distance of ", dist)


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

    #print('point 2', points[2])
    #print('point 4', points[4])
       
    if points[2] and points[4]:

        print('point 2', points[2])

        
        cv2.line(frame, points[2], points[4], (255,0, 0), 2)
        cv2.circle(frame, points[2], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[4], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        
       
       
        D = (dist.euclidean(points[2], points[4])/dpi)/centi
        (mX, mY) = midpoint(points[2], points[4])
        cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0, 255), 2)
        
    if points[5] and points[7]:
        
        cv2.line(frame, points[5], points[7], (255,0, 0), 2)
        cv2.circle(frame, points[5], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[7], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        
        D = (dist.euclidean(points[5], points[7])/dpi)/centi
        (mX, mY) = midpoint(points[5], points[7])
        cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0, 255), 2)
        
    if points[8] and points[10]:
        
        cv2.line(frame, points[8], points[10], (0,255, 0), 2)
        cv2.circle(frame, points[8], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[10], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        
        D = (dist.euclidean(points[8], points[10])/dpi)/centi
        (mX, mY) = midpoint(points[8], points[10])
        cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0, 255), 2)
        
    if points[11] and points[13]:
        
        cv2.line(frame, points[11], points[13], (0,255, 0), 2)
        cv2.circle(frame, points[11], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[13], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        
        D = (dist.euclidean(points[11], points[7])/dpi)/centi
        (mX, mY) = midpoint(points[13], points[7])
        cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0, 255), 2)
    
    if points[2] and points[11]:
        
        cv2.line(frame, points[2], points[11], (0,0, 255), 2)
        cv2.circle(frame, points[2], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(frame, points[11], 8, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
        
        D = (dist.euclidean(points[2], points[11])/dpi)/centi
        (mX, mY) = midpoint(points[2], points[11])
        cv2.putText(frame, "{:.1f} cm".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0, 0), 2)
        
   
cv2.imshow('Body_measuremnet', frame)
cv2.imwrite('Body_measurement.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

cv2.waitKey(0)
cv2.destroyAllWindows()


