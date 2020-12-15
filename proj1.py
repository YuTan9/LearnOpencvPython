import cv2 as cv2
import numpy as np

def color_detection(img, lower, upper):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower, upper)
    return mask

def trackbar():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    def empty(a):
        return a

    """16 26 195 255 124 232"""
    cv2.createTrackbar("H Min", "TrackBars", 16, 179, empty)
    cv2.createTrackbar("H Max", "TrackBars", 26, 179, empty)
    cv2.createTrackbar("S Min", "TrackBars", 195, 255, empty)
    cv2.createTrackbar("S Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("V Min", "TrackBars", 124, 255, empty)
    cv2.createTrackbar("V Max", "TrackBars", 232, 255, empty)

def get_val():
    lower = np.array([cv2.getTrackbarPos("H Min", "TrackBars"), cv2.getTrackbarPos("S Min", "TrackBars"),
                      cv2.getTrackbarPos("V Min", "TrackBars")])
    upper = np.array([cv2.getTrackbarPos("H Max", "TrackBars"), cv2.getTrackbarPos("S Max", "TrackBars"),
                      cv2.getTrackbarPos("V Max", "TrackBars")])
    return lower, upper

def get_contour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 50:
            # cv2.drawContours(imgResult, cont, -1, (255, 255, 255), 3)
            peri = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x+w//2, y+h//2

def draw(img, points):
    for point in points:
        cv2.circle(img, point, 15, (0, 255, 255), cv2.FILLED)

def main():
    cap = cv2.VideoCapture(1)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 150)
    trackbar()
    points = []
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        lower, upper = get_val()
        color = color_detection(img, lower, upper)
        x, y = get_contour(color)

        if x + y != 0:
            points.append((x, y))

        draw(img, points)

        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
