import cv2
import pickle
from datetime import datetime
import gen
import xlsxwriter

file_name = "Attendance.xlsx"
writer = xlsxwriter.Workbook(file_name)
worksheet = writer.add_worksheet()
namelist = []
video_capture = cv2.VideoCapture(0)
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("C:/python/attendence system/files/trainner.yml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascadePath)
labels = {"person_name": 1}
time = datetime.now()
row = 1
col = 0
padding = 20
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
gender_list = ['Male', 'Female']
with open("C:/python/attendence system/files/label.pickle", "rb") as f:
    og_lables = pickle.load(f)
    labels = {v: k for k, v in og_lables.items()}

while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minSize=(50, 50), minNeighbors=5)
    cv2.putText(frame, str(time.today()), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 240), 1, cv2.LINE_AA)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        riogray = gray[y:y+h, x:x+w]
        id_, conf = recogniser.predict(riogray)

        if conf >= 45 and conf <= 85:
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            colour = (0, 255, 0)
            font = cv2.FONT_HERSHEY_COMPLEX
            stock = 2
            p = 'p'
            gender = gen.gender(video_capture, gender_list, padding, MODEL_MEAN_VALUES)
            for i in range(100):
                if name not in namelist:
                    namelist.append(name)
                    date = time.strftime("%m/%d/%Y-%H:%M")
                    worksheet.write('A1', 'Name')
                    worksheet.write('B1', 'Gender')
                    worksheet.write('C1', 'p/a')
                    worksheet.write(row, col, name)
                    worksheet.write(row, col + 1, gender)
                    worksheet.write(row, col + 2, p)
                    row += 1
            cv2.putText(frame, name, (x, y), font, 0.6, colour, stock, cv2.LINE_AA)
    k = cv2.waitKey(1) & 0xff
    cv2.imshow(time.strftime("%a-%H:%M:%S"), frame)

    if k == ord('q'):
        break
writer.close()
video_capture.release()
cv2.destroyAllWindows()
