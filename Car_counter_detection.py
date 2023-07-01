import cv2
from ultralytics import YOLO
import math
import cvzone
from sort import *


# for real time
# cap = cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

#for video
cap = cv2.VideoCapture("../Videos/cars.mp4")



classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

model = YOLO("../YOLO-Weights/yolov8n.pt")

mask = cv2.imread("../Videos/mask.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3) #iou threshold is the confidence level of the bounding box to overlap the actual object

#max_age is the amound of frames the video has to wait for the object to be tracked again,so a value more than 15 is good for this project

limits = [400,297,673,297] #these are just the coordinated we are using to draw the line

totalCount =[]
while True:

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):      # this here will count the total no. of frames of the video and once it reaches the last one its set to restart the frames
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)       #this operation is used to put the mask(the black markings) on the img

    results = model(imgRegion, stream=True)     #used to only detect the region withing the mask

    detections = np.empty((0,5))       # first the array is going to be empty and is going to update with the object it gets updated
    #(0,5) is the format we are going to use ([x1,y1,x2,y2,conf)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = (x2 - x1), (y2 - y1)


            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass =="car" and conf > 0.4:     #here i have set the conf level >0.4 as it sometimes wolud also take incorrect detections and also maight take correct detections but with low conf level so this value has to be an average
                #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(5, x1), max(35, y1)), scale=1,thickness=1,offset=1)

                #cvzone.cornerRect(img, (x1, y1, w, h), l=10 ,rt=5)

                currentArray = np.array([x1,y1,x2,y2,conf])             #the empty array up will get automatically filled with the coordinates
                detections = np.vstack((detections,currentArray))       #we are stacking in the coordinates of the detections

    resultsT = tracker.update(detections)

    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)       #here the x coordinates are being set on with a particular y coordinate individually if the y coordinate differs then the line wont be straight

    for result in resultsT:
        x1, y1, x2, y2, id = result                                     #the detections are being given to result variable
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        #cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
        #                  scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2                                   #taken to find the center of the car
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)             #to put a circle in the center

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:             #here if the center passes bw the x-coordinate limits taken then it detects
                                                                                            # #but it also has to consider the y-coordinates i have taken the car to be detected a bit before it touches the actual line with the -15 and +15 so if a car is too fast it still gets detected

            if totalCount.count(id) == 0:                                                   #this is a if-else statement for if the the id is already detected it doesnt get detect again
                totalCount.append(id)                                                       #if the total cound of the id in the list is not zero then itll not get detected but if its the first time ie the total count is zero then it gets detected
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)#this will change the color of the line once a car passes it just gives more graphics

    cvzone.putTextRect(img, ("Total Cars"), (50, 80))
    cv2.putText(img, str(len(totalCount)), (380, 95), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8) #this here puts a text in which it takes out the total number of cars that passed by the length in the list function and it automatically updates with increasing values in the list

    cv2.imshow("IMAGE", img)

    cv2.waitKey(1)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



