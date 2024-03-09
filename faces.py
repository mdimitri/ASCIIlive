import cv2, time, sys, os, string
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread
import mediapipe

def savePic(pic):
    path = './Wall of fame'
    current_datetime = datetime.now()
    pathPic = os.path.join(path, current_datetime.strftime("%Y%m%d%H%M%S") + '.jpg')
    cv2.imwrite(pathPic, pic, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return

def apply_wiggly_pattern(image, frequency=10, amplitude=10, phase=0):
    rows, cols, _ = image.shape

    # Generate wiggly distortion map
    x = np.arange(cols)
    y = np.arange(rows)
    x_distortion = amplitude * np.sin(2 * np.pi * frequency * x / cols + phase)
    y_distortion = amplitude * np.sin(2 * np.pi * frequency * y / rows + phase)
    x_distortion_map, y_distortion_map = np.meshgrid(x_distortion, y_distortion)

    # Create distorted indices
    distorted_x = np.clip(np.arange(cols) + x_distortion_map[0, :], 0, cols - 1).astype(np.float32)
    distorted_y = np.clip(np.arange(rows) + y_distortion_map[:, 0], 0, rows - 1).astype(np.float32)

    # Create meshgrid from distorted indices
    distorted_x_map, distorted_y_map = np.meshgrid(distorted_x, distorted_y)

    # Interpolate values using remap
    distorted_image = cv2.remap(image, distorted_x_map, distorted_y_map, interpolation=cv2.INTER_LINEAR)

    return distorted_image

    return distorted_image
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

    drawSpecOval     = mpDraw.DrawingSpec(thickness=6, circle_radius=0, color=(255, 255, 255))
    drawSpecEyeBrows = mpDraw.DrawingSpec(thickness=6, circle_radius=0, color=(255, 255, 255))
    drawSpecEyes     = mpDraw.DrawingSpec(thickness=6, circle_radius=0, color=(255, 255, 255))
    drawSpecLips     = mpDraw.DrawingSpec(thickness=6, circle_radius=0, color=(255, 255, 255))
    drawSpecNose     = mpDraw.DrawingSpec(thickness=4, circle_radius=0, color=(255, 255, 255))
    drawSpecMesh     = mpDraw.DrawingSpec(thickness=1, circle_radius=0, color=(220, 220, 220))

    cv2.namedWindow('faces', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('faces', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while True:
        # Read the latest frame from the webcam
        # if frameNo>0:
        #     oldFrame = frame
        current_time = datetime.now()
        seconds = current_time.second + current_time.microsecond / 1000000

        frame = np.ascontiguousarray(camera.frame)

        # make a trippy background
        background = cv2.cvtColor(camera.frame, cv2.COLOR_BGR2HSV)
        # boost colors, and rotate hue over time

        background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        background[:, :, 0] = np.mod(background[:, :, 0] + seconds / (10 / (180)), 180)
        background[:, :, 1] = ((background[:, :, 1] / 255.0) ** 0.1) * 255
        background[:, :, 2] = ((background[:, :, 2] / 255.0) ** 0.3) * 255
        background = cv2.cvtColor(background, cv2.COLOR_HSV2BGR)

        # frameNP = np.ascontiguousarray(frame)
        results = faceMesh.process(frame)

        frame *= 0

        if results.multi_face_landmarks:
            # draw faces
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







        # # Find the contours in the binary image
        # contours, hierarchy = cv2.findContours(frame[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # # Create a blank image with the same dimensions as the original image
        # filled_img = np.zeros(frame.shape[:2], dtype=np.uint8)
        # # Iterate over the contours and their hierarchies
        # for i, contour in enumerate(contours):
        #     # If the contour doesn't have a parent, fill it with pixel value 255
        #     cv2.drawContours(filled_img, [contour], -1, 255, cv2.FILLED)
        # filled_img = np.repeat(filled_img[:, :, np.newaxis], 3, axis=2)

        # background[filled_img==255] =  cv2.GaussianBlur(background, (11, 11), 0)[filled_img==255]
        # background = cv2.resize(background, (background.shape[1]//16, background.shape[0]//16), interpolation=cv2.INTER_NEAREST)
        # background = cv2.resize(background, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


        phase = np.deg2rad(np.mod(seconds / (1 / (360)), 360))
        background = apply_wiggly_pattern(background, frequency=5, amplitude=10, phase=phase)

        canvasBlend = np.where(frame == (0, 0, 0), background, frame)

        canvasBlend = cv2.resize(canvasBlend, (1920, 1200), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('faces', cv2.cvtColor(canvasBlend, cv2.COLOR_BGR2RGB))
        key = cv2.waitKey(1)
        if key == ord('x'):
            return
        elif key == ord(' '):
            savePic(canvasBlend)
        continue

    return

if __name__ == '__main__':
    main()