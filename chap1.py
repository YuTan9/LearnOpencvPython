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

"""read and show an image"""
# img = cv2.imread("lena.jpg")
# cv2.imshow("output", img)
# cv2.waitKey(0)

"""Show image captured by web cam in realtime"""
# cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
#
# while True:
#     success, img = cap.read()
#     cv2.imshow("Webcam", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

"""Basic functions"""
# img = cv2.imread("lena.jpg")
"""color convert"""
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", imgGray)
# cv2.waitKey(2000)

"""blur image"""
# imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
# cv2.imshow("Blur", imgBlur)
# cv2.waitKey(2000)

"""canny edge detection"""
# imgCanny = cv2.Canny(img, 200, 200)
# cv2.imshow("Edge", imgCanny)
# cv2.waitKey(2000)

"""dilate"""
# kernel = np.ones((5, 5), np.uint8)
# imgCanny = cv2.Canny(img, 200, 200)
# imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
# cv2.imshow("Dilate", imgDilation)
# cv2.imshow("Edge", imgCanny)
# cv2.waitKey(0)

"""erode"""
# kernel = np.ones((5, 5), np.uint8)
# imgCanny = cv2.Canny(img, 200, 200)
# imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)
# imgErode = cv2.erode(imgDilation, kernel, iterations = 1)
# cv2.imshow("Dilate", imgDilation)
# cv2.imshow("Edge", imgCanny)
# cv2.imshow("Eroded", imgErode)
# cv2.waitKey(0)

"""Resize"""
# print(img.shape)
#
# imgResize = cv2.resize(img, (100, 100))
# cv2.imshow("Image Resize",imgResize)
# cv2.waitKey(0)

"""Crop"""
# print(img.shape)
# imgCropped = img[0:110, 0:220]
# cv2.imshow("Image", img)
# cv2.imshow("Image Cropped", imgCropped)
# cv2.waitKey(0)

"""draw lines"""
# img = np.zeros((512, 512, 3), np.uint8)
# # img[:] = 255, 0, 0
# cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
# cv2.rectangle(img, (0, 0), (100, 160), (0, 0, 255))
# cv2.rectangle(img, (100, 160), (200, 200), (100, 100, 100), cv2.FILLED)
# cv2.circle(img, (450, 450), 20, (255, 255, 255))
# cv2.putText(img, "Hello World", (300, 500), cv2.FONT_ITALIC, 1, (0, 100, 300), 3)
# cv2.imshow("image", img)
# cv2.waitKey(0)

"""Warp perspective"""
# img = cv2.imread("cards.jpg")
#
# width, height = 250, 350
# pts1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
# pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# matrix = cv2.getPerspectiveTransform(pts1, pts2)
# imgOutput = cv2.warpPerspective(img, matrix, (width, height))
#
# cv2.imshow("Image", img)
# cv2.imshow("Output", imgOutput)
# cv2.waitKey(0)

"""Join images"""
# img = cv2.imread("lena.jpg")
# hor = np.hstack((img, img))
# vert= np.vstack((img, img))
# cv2.imshow("Horizontal", hor)
# cv2.imshow("Vertical", vert)
# cv2.waitKey(0)

"""Color detection"""
# def empty(a):
#     pass
#
# img = cv2.imread("Color.jpg")
# imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",0,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
#
#
# while True:
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     # print(h_min, h_max, s_min, s_max, v_min, v_max)
#     lower, upper = np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
#     mask = cv2.inRange(imgHSV, lower, upper)
#     # mask is in {0, 255}, if pixel is in lower, upper bounds, that pixel is 255(white), else 0(black)
#     imgResult = cv2.bitwise_and(img, img, mask = mask)
#
#     cv2.imshow("HSV", imgHSV)
#     cv2.imshow("Mask", mask)
#     cv2.imshow("Result", np.vstack((img, imgResult)))
#     cv2.waitKey(1)

"""Shape detection"""
# def getContours(img):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cv2.putText(imgCont, "Area, perimeter, corners of shape", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 100, 0), 2)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500: # filter noise
#             cv2.drawContours(imgCont, cnt, -1, (0, 0, 0), 3)
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#             obj_cor = len(approx)
#             # objCor of circle is not 0 but high value (gets 8 for this example)
#             x, y, w, h = cv2.boundingRect(approx)
#             text = str(int(area)) + ", " + str(int(peri)) + ", " + str(int(obj_cor))
#             cv2.putText(imgCont, text, (x, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 100, 0), 2)
#
# img = cv2.imread("shapes.png")
# imgCont = img.copy()
# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# imgBlur = cv2.blur(imgGray, (3, 3))
# imgCanny= cv2.Canny(imgBlur, 50, 50)
# getContours(imgCanny)
# imgStack = stackImages(0.6, ([img, imgGray, imgBlur], [imgBlur, imgCanny, imgCont]))
# cv2.imshow("", imgStack)
#
# cv2.waitKey(0)


"""Face detection"""
faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
img = cv2.imread("image/lena.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# webcam = cv2.VideoCapture(1)
#
# while True:
#     success, frame = webcam.read()
#     frame = cv2.flip(cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))), 1)
#     frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(frameGray, 2, 5)
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     cv2.imshow("Webcam", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("Faces", img)
cv2.waitKey(0)








