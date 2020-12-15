import cv2 as cv2
import numpy as np

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

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (21, 21), 1)
    img_edge = cv2.Canny(img_blur, 200, 200)
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_edge, kernel , iterations = 1)
    img_threshold = cv2.erode(img_dial, kernel, iterations = 1)
    return img_threshold

def contour(img):
    biggest = np.array([])
    max_area = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cont in contours:
        area = cv2.contourArea(cont)
        if cv2.contourArea(cont) > 5000:
            peri = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest

def reorder(points):
    points = points.reshape((4,2))
    new_points = np.zeros((4, 1, 2), np.int32)

    # sum the x, y positions
    add = points.sum(1)
    # the smallest of add is assigned to the first and the largest is at the end
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]

    #find the difference in x, y position
    diff = np.diff(points, axis = 1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def warp(img, points):
    points = reorder(points)
    width, height = points[3][0][0] - points[0][0][0], points[3][0][1] - points[0][0][1]
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_wrap = cv2.warpPerspective(img, matrix, (width, height))
    return img_wrap

def main():
    img = cv2.imread("image/doc.jpg")
    img_threshold = preprocess(img)
    points = contour(img_threshold)

    img_output = np.zeros((640, 480, 3), np.uint8)
    if len(points) > 0:
        # for point in points:
        #     pos = (point[0][0], point[0][1])
        #     cv2.circle(img, pos, 30, (255, 0, 0), cv2.FILLED)
        img_output = warp(img, points)
        cv2.imwrite("Document.jpg", img_output)
        img_output = cv2.resize(img_output, (int(img_output.shape[1]*0.3), int(img_output.shape[0]*0.3)))


    stack = stackImages(0.2, [img, img_output])
    cv2.imshow("Result", img_output)
    cv2.imshow("Image", stack)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()