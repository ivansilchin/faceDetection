from mtcnn.mtcnn import MTCNN
import cv2


def update_frame(frame_to_update):
    faces = detector.detect_faces(frame)
    for face in faces:
        (x, y, width, height) = face['box']
        cv2.rectangle(frame_to_update, (x, y), (x + width, y + height), (255, 255, 255), 1)
        radius = max(min(width, height) // 50, 1)
        for keypoint in face['keypoints']:
            cv2.circle(frame_to_update, face['keypoints'][keypoint], radius, (255, 255, 255), -1)
    return frame_to_update


vc = cv2.VideoCapture(0)
cv2.namedWindow('detector')

if vc.isOpened():
    (rval, frame) = vc.read()
else:
    rval = False

detector = MTCNN()

while rval:
    frame = update_frame(frame)
    cv2.imshow('detector', frame)
    rval, frame = vc.read()
    if cv2.waitKey(1) != -1:
        break

vc.release()
cv2.destroyWindow('detector')
