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
cap = cv2.VideoCapture("../Videos/people.mp4")



classNames = ["person", "bicycle", "car", "motorcycles", "aeroplane", "bus", "train", "truck", "boat",
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

mask=cv2.imread("../Videos/mask2.png")

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits=[210,400,380,370]
limitss=[460,400,650,370]
totalCount =[]
totalCountt =[]
while True:
    success, img = cap.read()

    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

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

            if currentClass =="person" and conf > 0.3:

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    resultsT = tracker.update(detections)


    for result in resultsT:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1

        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        cv2.line(img, (limitss[0], limitss[1]), (limitss[2], limitss[3]), (0, 255, 0), 5)

        cx,cy = x1+w//2 , y1+h//2
        cv2.circle(img,(cx,cy), 5,(255,255,0),cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:


            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0),5)

        if limitss[0] < cx < limitss[2] and limitss[1] - 15 < cy < limitss[1] + 15:


            if totalCountt.count(id) == 0:
                totalCountt.append(id)
                cv2.line(img, (limitss[0], limitss[1]), (limitss[2], limitss[3]), (0, 255, 0),5)


    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.putText(img, str(len(totalCountt)), (400, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    cv2.imshow("IMAGE", img)

    cv2.waitKey(4)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



