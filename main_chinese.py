import cv2
import time
from msvcrt import getch
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Detect faces and reconstruct face bounding boxes
def getfacebox(net, frame, conf_threshold=0.0):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    #  blobFromImage(image[, scalefactor[, size[, mean[, swapRB[, crop[, ddepth]]]]]]) -> retval  返回值   # swapRB是交换第一个和最后一个通道   返回按NCHW尺寸顺序排列的4 Mat值
    net.setInput(blob)
    detections = net.forward()  # face
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # print(confidence)
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # bounding box 的坐标
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)  # rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
        break
    return frameOpencvDnn, bboxes

# net model & Pre-training model
faceProto = "./models/opencv_face_detector.pbtxt"
faceModel = "./models/opencv_face_detector_uint8.pb"

ageProto = "./models/age_deploy.prototxt"
ageModel = "./models/age_net.caffemodel"

genderProto = "./models/gender_deploy.prototxt"
genderModel = "./models/gender_net.caffemodel"

# model mean
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['男性', '女性']

# Loading network
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
# Face detection network and model
faceNet = cv2.dnn.readNet(faceModel, faceProto)

# Open a video file or a picture or a camera
camera = cv2.VideoCapture(0)

frame_width = int(camera.get(3))
frame_height = int(camera.get(4))
out = cv2.VideoWriter('./outputtest.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

padding = 20
# print("befor while")
while cv2.waitKey(1) < 0:
    # Read frame
    t = time.time()
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        print("exit")
        break    
    # print(faceNet)
    
    frameFace, bboxes = getfacebox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)   # get face image
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
        print("=======", type(face), face.shape)  #  <class 'numpy.ndarray'> (166, 154, 3)
        #
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        print("======", type(blob), blob.shape)  # <class 'numpy.ndarray'> (1, 3, 227, 227)
        genderNet.setInput(blob)   # Blob input network for gender detection
        genderPreds = genderNet.forward()   # Gender detection for forward propagation
        print("++++++", type(genderPreds), genderPreds.shape, genderPreds)   # <class 'numpy.ndarray'> (1, 2)  [[9.9999917e-01 8.6268375e-07]]  变化的值
        gender = genderList[genderPreds[0].argmax()]   # return gender
        # print("Gender Output : {}".format(genderPreds))
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(agePreds[0].argmax())  # 3
        print("*********", agePreds[0])   #  [4.5557402e-07 1.9009208e-06 2.8783199e-04 9.9841607e-01 1.5261240e-04 1.0924522e-03 1.3928890e-05 3.4708322e-05]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

        # label = "{},{}".format(gender, age) # include gender and age
        if age in ['(0-2)', '(4-6)', '(8-12)']:
            flag = "Child"
        else:
            flag = "Adult"

        ## Use simsum.ttc to write Chinese.
        b,g,r,a = 0,255,0,0
        fontpath = "./simsun.ttc" # <== Chinese ttc location
        font = ImageFont.truetype(fontpath, 32)
        img_pil = Image.fromarray(frameFace)
        draw = ImageDraw.Draw(img_pil)
        draw.text((bbox[0], bbox[1] - 10),  f"{gender},{age}", font = font, fill = (b, g, r, a))
        frameFace = np.array(img_pil)

        # cv2.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
        cv2.imshow("Age Gender Demo", frameFace)
        out.write(frameFace)
        
    print("time : {:.3f} ms".format(time.time() - t))

# When everything done, release the video capture and video write objects
camera.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()