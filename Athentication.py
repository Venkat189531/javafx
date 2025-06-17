import cv2
import numpy as np
import face_recognition
import os

# Load images and encode known faces
path = 'Images'
images = []
classNames = []
myList = os.listdir(path)

for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeList.append(encodes[0])
    return encodeList

encodingListKnown = findEncodings(images)
print("Encoding complete!")

def scan_and_check_face():
    cap = cv2.VideoCapture(0)
    result = False

    while True:
        success, frame = cap.read()
        if not success:
            break

        smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgbSmallFrame = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)

        faceLocations = face_recognition.face_locations(rgbSmallFrame)
        encodeCurrent = face_recognition.face_encodings(rgbSmallFrame, faceLocations)

        if len(encodeCurrent) > 0:
            encodeFace = encodeCurrent[0]

            matches = face_recognition.compare_faces(encodingListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodingListKnown, encodeFace)

            if len(faceDis) > 0:
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()

                    print(f"Recognized: {name}")
                    result = True
                    break
                else:
                    print("Face detected but not recognized.")
                    result = False
                    break

        cv2.imshow('Face Scan', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit requested.")
            result = False
            break

    cap.release()
    cv2.destroyAllWindows()
    return result

# Run the function
if __name__ == "__main__":
    recognized = scan_and_check_face()
    print(f"Result: {recognized}")
