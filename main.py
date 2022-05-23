import cv2
import dlib
import time
import math
import vehicles

carCascade = cv2.CascadeClassifier('vech.xml')
video = cv2.VideoCapture('carsVid.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 14
    speed = d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    Vehicle_Tracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000

    out = cv2.VideoWriter('outNew.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        rc, image = video.read()
        if type(image) == type(None):
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for VehicleID in Vehicle_Tracker.keys():
            trackingQuality = Vehicle_Tracker[VehicleID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(VehicleID)

        for VehicleID in carIDtoDelete:
            Vehicle_Tracker.pop(VehicleID, None)
            carLocation1.pop(VehicleID, None)
            carLocation2.pop(VehicleID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h

                matchCarID = None

                for VehicleID in Vehicle_Tracker.keys():
                    trackedPosition = Vehicle_Tracker[VehicleID].get_position()

                    X_Tracker = int(trackedPosition.left())
                    Y_Tracker = int(trackedPosition.top())
                    Width_Tracker = int(trackedPosition.width())
                    Height_Tracker = int(trackedPosition.height())

                    t_x_bar = X_Tracker + 0.5 * Width_Tracker
                    t_y_bar = Y_Tracker + 0.5 * Height_Tracker

                    if ((X_Tracker <= x_bar <= (X_Tracker + Width_Tracker)) and (Y_Tracker <= y_bar <= (Y_Tracker + Height_Tracker)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = VehicleID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    Vehicle_Tracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for VehicleID in Vehicle_Tracker.keys():
            trackedPosition = Vehicle_Tracker[VehicleID].get_position()

            X_Tracker = int(trackedPosition.left())
            Y_Tracker = int(trackedPosition.top())
            Width_Tracker = int(trackedPosition.width())
            Height_Tracker = int(trackedPosition.height())

            cv2.rectangle(resultImage, (X_Tracker, Y_Tracker), (X_Tracker + Width_Tracker, Y_Tracker + Height_Tracker), rectangleColor, 2)

            carLocation2[VehicleID] = [X_Tracker, Y_Tracker, Width_Tracker, Height_Tracker]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and 275 <= y1 <= 285:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(x1 + w1 / 2), int(y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    trackMultipleObjects()
