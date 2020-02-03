import cv2
import sys
import os.path
from glob import glob
count = 0

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    global count
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (128, 128))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("faces/" + save_filename, face)
        count += 1
        print(count)


if __name__ == '__main__':
    if os.path.exists('faces') is False:
        os.makedirs('faces')
    file_list = glob('*.jpg')
    for filename in file_list:
        detect(filename)
    file_list = glob('*.png')
    for filename in file_list:
        detect(filename)