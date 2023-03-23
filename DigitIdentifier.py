import cv2
import numpy as np

from Model import get_trained_model
print("Initializing the project:")

model = get_trained_model(training=False)

print("Select the mode:\n\tPress 1 for webcam\n\tPress 2 for playing from video clip\n")

x = int(input())

## Name of the window
cvWin_name = "Moxank's (1064) Output"


vd = None
if x == 1:
    vd = cv2.VideoCapture(0)
else:
    pth = input("Enter the path to video: ")
    vd = cv2.VideoCapture(pth)
#vd.set(cv2.CAP_PROP_FPS, 10)
kernel = np.ones((3, 3))
while True:
    ret, fr_o = vd.read()
    if not ret or fr_o is None:
        continue

    grayscale_img = cv2.cvtColor(fr_o, cv2.COLOR_BGR2GRAY)

    edged_img = cv2.Canny(grayscale_img, 50, 150)

    contours, _ = cv2.findContours(
        edged_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_box = []

    image_cntrs = cv2.drawContours(cv2.cvtColor(
        edged_img, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2)

    avg_width, avg_len = 0, 0
    for cntr in contours:
        box = list(cv2.boundingRect(cntr))
        avg_width, avg_len = avg_width+box[2], avg_len+box[3]
        rect_box.append(box)

    avg_width /= len(contours) + 1
    avg_len /= len(contours) + 1

    avg_width *= 0.3
    avg_len *= 0.6

    nrect_box = []

    for box in rect_box:
        if box[2] and box[3]:
            if box[2] > avg_width and box[3] > avg_len and box[2] < (10*avg_width/0.3) and box[3] < (10*avg_width/0.6):
                nrect_box.append(box)
    rect_box = nrect_box.copy()
    nrect_box = []

    vis_box = [0]*len(rect_box)
    for i in range(len(rect_box)):
        box1 = rect_box[i]
        if vis_box[i] == 0:
            for j in range(i+1, len(rect_box)):
                if vis_box[j] == 0:
                    box2 = rect_box[j]

                    top_left_crns = (
                        max(box1[0], box2[0]), max(box1[1], box2[1]))
                    btm_rgth_crns = (
                        min(box1[0]+box1[2], box2[0]+box2[2]), min(box1[1]+box1[3], box2[1]+box2[3]))

                    w, l = max(
                        0, btm_rgth_crns[0] - top_left_crns[0]), max(0, btm_rgth_crns[1]-top_left_crns[1])
                    intersect_area = w*l
                    union_area = (box1[2]*box1[3]) + \
                        (box2[2]*box2[3]) - intersect_area
                    iou = intersect_area/max(union_area, 1)

                    if iou > 0.09:
                        vis_box[j] = 1
                        box1[0], box1[1] = min(box1[0], box2[0]), min(box1[1], box2[1])
                        box1[2], box1[3] = max(box1[2], box2[2]), max(box1[3], box2[3])
            nrect_box.append(box1)

    
    fimg = np.ones(fr_o.shape[:2], dtype=np.uint8) * 255

    #############
    max_y = 0
    cur_x, cur_y = 10, 10



    fr = fr_o.copy()

    predict_images = []
    img_dims = []
    for x, y, w, h in nrect_box:
        if x+w >= fr_o.shape[1] or y+h >= fr_o.shape[0]:
            continue
        sq_img = np.zeros((max(w,h),max(w,h)),dtype=np.uint8) 
        dxw,dxh = 0,0
        diff = abs(w-h)
        if w < h:
            dxw += (diff//2)
        else:
            dxh += (diff//2)
        sq_img[dxh:h+dxh,dxw:w+dxw] = edged_img[y:y+h, x:x+w]

        sq_img = np.pad(sq_img,int(w*0.3))
        original_crd = (x,y,w,h)
        
        
        w,h = sq_img.shape
        kernel_dilate = np.ones((11,11))
        sq_img = cv2.dilate(sq_img/255,kernel,1)*255
        resized_img = cv2.resize(sq_img,(28,28),interpolation = cv2.INTER_AREA)
        resized_img = np.expand_dims(resized_img,axis = -1)/255
        
        

        predict_images.append(resized_img)
        img_dims.append(original_crd)

        if cur_y + h >= fimg.shape[0]:
            continue
        if cur_x + w >= fimg.shape[1]:
            max_y = cur_y
            cur_x = 10
        else:
            fimg[max_y:max_y + h, cur_x:cur_x+ w] = sq_img
            cur_x += w + 10
            cur_y = max(cur_y, max_y+h+10)
    if len(predict_images):
        predictions = np.argmax(model.predict(np.array(predict_images),verbose = 0),axis = 1)
        for num_predicted,(x,y,w,h) in zip(predictions,img_dims):
            fr = cv2.rectangle(fr, (x, y), (x+w, y+h), (0, 255, 0), 3)
            #COLOR: #317773
            fr = cv2.putText(fr,str(num_predicted), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0x31,0x77, 0x73), 5)


    l1 = np.hstack((fr_o, cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)))
    l2 = np.hstack((cv2.cvtColor(edged_img, cv2.COLOR_GRAY2BGR), image_cntrs))
    l3 = np.hstack((cv2.cvtColor(fimg, cv2.COLOR_GRAY2BGR), fr))
    l123 = np.vstack((l1, l2, l3))
    cv2.imshow(cvWin_name, l123)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()