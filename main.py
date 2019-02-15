import numpy as np
import cv2
from PIL import  Image
import ImageHeader
import Utility
import NeuralNet
import random

def main():



    face_cascade = cv2.CascadeClassifier("./haar_cascade/haarcascade_frontalface_default.xml")

    imgHdr = ImageHeader.ImageHeader()
    imgHdr.maxImages = 165
    imgHdr.imgWidth = 320
    imgHdr.imgHeight = 243
    nn = NeuralNet.NeuralNet(imgHdr)
    ut = Utility.Utility()
    dirname = "./yale/yalefaces"
    filename = ut.readFileLabel(dirname)

    for ii in range(len(filename)):
        img = ut.readGIFFile("./yale/yalefaces/" + filename[ii], imgHdr)
        nn.getTrainImg(ii, filename[ii], img)

        randomNum = random.randint(0,165)
        test_img = ut.readGIFFile("./yale/yalefaces/" + filename[randomNum], imgHdr)
        nn.getTestImg(ii, filename[randomNum], test_img)

    nn.trainMNIST()
        # faces = face_cascade.detectMultiScale(img, 1.3, 5)
        # for x, y, w, h in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #
        # gabor_filter = cv2.getGaborKernel((imgHdr.imgWidth, imgHdr.imgHeight), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        # filtered_img = cv2.filter2D(img, cv2.CV_8UC3, gabor_filter)
        # cv2.imshow("GIF Image", img)
        # cv2.imshow("Filtered Image", filtered_img)
        # cv2.waitKey()



if __name__ == '__main__':
    main()
