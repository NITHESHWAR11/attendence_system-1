import cv2
import argparse


def highlight_face(net, frame, conf_threshold=0.7):
    frame_open_dnn = frame.copy()
    frame_height = frame_open_dnn.shape[0]
    frame_width = frame_open_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_open_dnn, 1.0, (300, 300), [140, 177, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3]*frame_width)
            y1 = int(detections[0, 0, i, 4]*frame_width)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_width)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_open_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height/150)), 8)
    return frame_open_dnn, face_boxes


par_set = argparse.ArgumentParser()
par_set.add_argument('--images')
args = par_set.parse_args()

face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
gender_proto = "gender_deploy.prototxt"
gender_model = "gender_net.caffemodel"

face_net = cv2.dnn.readNet(face_model, face_proto)
gender_NET = cv2.dnn.readNet(gender_model, gender_proto)


def gender(video, gender_list, padding, model_mean_values):
        hasFrame, frame = video.read()
        result_img, face_boxer = highlight_face(face_net, frame)
        for face_box in face_boxer:
            face = frame[max(0, face_box[1] - padding):
                         min(face_box[3]+padding, frame.shape[0]-1), max(0, face_box[0] - padding):
                         min(face_box[2]+padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
            gender_NET.setInput(blob)
            gender_preds = gender_NET.forward()
            gender = gender_list[gender_preds[0].argmax()]

            return gender

