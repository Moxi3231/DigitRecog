import cv2
import numpy as np

print("Select the mode:\n\tPress 1 for webcam\n\tPress 2 for playing from video clip\n")
x = int(input())
vd = None
if x == 1:
    vd = cv2.VideoCapture(0)
else:
    pth = input("Enter the path to video: ")
    vd = cv2.VideoCapture(pth)

kernel = np.ones((3,3))
while True:
    _,fr_o = vd.read()
    if fr_o is None:
        continue
    grayscale_img = cv2.cvtColor(fr_o,cv2.COLOR_BGR2GRAY)
    
    edged_img = cv2.Canny(grayscale_img,50,150)

    contours,_ = cv2.findContours(edged_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rect_box = []

    avg_width,avg_len = 0,0
    for cntr in contours:
        box = list(cv2.boundingRect(cntr))
        avg_width,avg_len = avg_width+box[2],avg_len+box[3]
        rect_box.append(box)
        
    avg_width /= len(contours) + 1
    avg_len /= len(contours) + 1
    
    
    avg_width *= 0.3
    avg_len *= 0.6

    nrect_box = []

    for box in rect_box:
        if box[2] > avg_width and box[3] > avg_len:
            nrect_box.append(box)
    rect_box = nrect_box.copy()
    nrect_box = []


    vis_box = [0]*len(rect_box)
    for i in range(len(rect_box)):
        box1 = rect_box[i]
        if vis_box[i] == 0:
            for j in range(i+1,len(rect_box)):
                if vis_box[j] == 0:
                    box2 = rect_box[j]

                    top_left_crns = (max(box1[0],box2[0]),max(box1[1],box2[1]))
                    btm_rgth_crns = (min(box1[0]+box1[2],box2[0]+box2[2]),min(box1[1]+box1[3],box2[1]+box2[3]))

                    w,l = max(0,btm_rgth_crns[0] - top_left_crns[0]),max(0,btm_rgth_crns[1]-top_left_crns[1])
                    intersect_area = w*l
                    union_area = (box1[2]*box1[3]) + (box2[2]*box2[3]) - intersect_area
                    iou = intersect_area/max(union_area,1)

                    if iou > 0.09:
                        vis_box[j] = 1
                        box1[0],box1[1] = min(box1[0],box2[0]),min(box1[1],box2[1])
                        box1[2],box1[3] = max(box1[2],box2[2]),max(box1[3],box2[3])
            nrect_box.append(box1)

    fr = fr_o.copy()
    for x,y,w,h in nrect_box:
        fr = cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),3)
        fr = cv2.putText(fr,"dsa",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),1)
    #fr = cv2.drawContours(fr_o,contours,-1,(0,255,0),3)

    l1 = np.hstack((fr_o,cv2.cvtColor(grayscale_img,cv2.COLOR_GRAY2BGR)))
    l2 = np.hstack((cv2.cvtColor(edged_img,cv2.COLOR_GRAY2BGR),fr))
    l12 = np.vstack((l1,l2))
    cv2.imshow('Output',l12)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break