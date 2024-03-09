import cv2, time, sys, os, string
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread
import mediapipe

def cropFrame(image, crop_dimension=200):
    height, width = image.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = width / height

    # Calculate the crop size for the other dimension to maintain the aspect ratio
    if width > height:
        crop_height = int(crop_dimension / aspect_ratio)
        crop_width = crop_dimension
    else:
        crop_width = int(crop_dimension * aspect_ratio)
        crop_height = crop_dimension

    # Calculate the coordinates for the top-left and bottom-right corners of the crop
    top_left_x = (width - crop_width) // 2
    top_left_y = (height - crop_height) // 2
    bottom_right_x = top_left_x + crop_width
    bottom_right_y = top_left_y + crop_height

    # Crop the center portion of the image
    cropped_image = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    return cropped_image

class RGBcamera(Thread):
    def __init__(self, targetResolution, cropSize):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, targetResolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, targetResolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.targetResolution = targetResolution
        self.cropSize = cropSize

        # read in one frame
        self.frame = self.cap.read()[1]
        if self.frame.shape[0] < self.targetResolution[1]:
            cutWidth = int((self.frame.shape[1] - self.frame.shape[0] * self.targetResolution[0] / self.targetResolution[1]) / 2)
            self.frame = self.frame[:, cutWidth: -cutWidth, :]
        self.frame = np.flip(self.frame, axis=1)
        # crop central portion of grabbed frame
        self.frame = cropFrame(self.frame, crop_dimension=self.cropSize)
        # start async
        self.start()


    def run(self):
        while True:
            self.frame = self.cap.read()[1]
            if self.frame.shape[0] < self.targetResolution[1]:
                cutWidth = int((self.frame.shape[1] - self.frame.shape[0] * self.targetResolution[0] / self.targetResolution[1]) / 2)
                self.frame = self.frame[:, cutWidth: -cutWidth, :]
            self.frame = np.flip(self.frame, axis=1)
            # crop central portion of grabbed frame
            self.frame = cropFrame(self.frame, crop_dimension=self.cropSize)
def main():
    # setting for the webcam
    targetResolution = [1280 // 1, 800 // 1]
    # crop the central portion
    cropSize = int(targetResolution[0] / 1.5)

    camera = RGBcamera(targetResolution=targetResolution, cropSize=cropSize)
    cv2.waitKey(100)

    mpDraw = mediapipe.solutions.drawing_utils
    mpFaceMesh = mediapipe.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=2)
    drawSpecOval     = mpDraw.DrawingSpec(thickness=2, circle_radius=0, color=(255, 255, 255))
    drawSpecEyeBrows = mpDraw.DrawingSpec(thickness=5, circle_radius=0, color=(192, 128, 64))
    drawSpecEyes     = mpDraw.DrawingSpec(thickness=2, circle_radius=0, color=(0, 64, 255))
    drawSpecLips     = mpDraw.DrawingSpec(thickness=4, circle_radius=0, color=(255, 0, 0))
    drawSpecNose     = mpDraw.DrawingSpec(thickness=2, circle_radius=0, color=(255, 255, 255))
    drawSpecMesh     = mpDraw.DrawingSpec(thickness=1, circle_radius=0, color=(192, 192, 192))

    cv2.namedWindow('faces', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('faces', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        # Read the latest frame from the webcam
        # if frameNo>0:
        #     oldFrame = frame

        frame = np.ascontiguousarray(camera.frame)

        # make a trippy background
        background = cv2.cvtColor(camera.frame, cv2.COLOR_BGR2HSV)
        background[:,:,0] +=

        # frameNP = np.ascontiguousarray(frame)
        results = faceMesh.process(frame)
        frameBackground = np.copy(frame)
        frame *= 0

        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecMesh)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_NOSE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecNose)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_FACE_OVAL,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecOval)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LIPS,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecLips)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_RIGHT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecEyes)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LEFT_EYE,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecEyes)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_RIGHT_EYEBROW,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecEyeBrows)
                mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LEFT_EYEBROW,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=drawSpecEyeBrows)

        cv2.imshow('faces', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        continue

    return

if __name__ == '__main__':
    main()