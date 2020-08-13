import cv2 as cv2
import numpy as np
print("Package imported")
"""Handy function defined by Murtaza"""
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
# def empty(a):
#     pass
#
# img = cv2.imread("Color.jpg")
# imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("B Min","TrackBars",62,255,empty)
# cv2.createTrackbar("B Max","TrackBars",246,255,empty)
# cv2.createTrackbar("G Min","TrackBars",123,255,empty)
# cv2.createTrackbar("G Max","TrackBars",255,255,empty)
# cv2.createTrackbar("R Min","TrackBars",0,255,empty)
# cv2.createTrackbar("R Max","TrackBars",189,255,empty)
#
#
# while True:
#     b_min = cv2.getTrackbarPos("B Min", "TrackBars")
#     b_max = cv2.getTrackbarPos("B Max", "TrackBars")
#     g_min = cv2.getTrackbarPos("G Min", "TrackBars")
#     g_max = cv2.getTrackbarPos("G Max", "TrackBars")
#     r_min = cv2.getTrackbarPos("R Min", "TrackBars")
#     r_max = cv2.getTrackbarPos("R Max", "TrackBars")
#     # print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower, upper = np.array([b_min, g_min, r_min]), np.array([b_max, g_max, r_max])
#     mask = cv2.inRange(img, lower, upper)
#     # mask is in {0, 255}, if pixel is in lower, upper bounds, that pixel is 255(white), else 0(black)
#     imgResult = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
#
#     # cv2.imshow("HSV", imgHSV)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Result", np.vstack((img, imgResult)))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# def empty(a):
#     print(a)
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("B Min","TrackBars",0,100,empty)
# cv2.waitKey(0)