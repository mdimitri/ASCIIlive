import cv2, time, sys, os, string
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread
import mediapipe as mp

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def savePic(pic):
    path = './Wall of fame'
    current_datetime = datetime.now()
    pathPic = os.path.join(path, current_datetime.strftime("%Y%m%d%H%M%S") + '.jpg')
    cv2.imwrite(pathPic, pic, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes, fig, ax):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  ax.cla()
  ax.set_xlim(0, 1)

  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.pause(0.0001)
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

    base_options = python.BaseOptions(model_asset_path='./weights/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    fig, ax = plt.subplots(figsize=(12, 12))

    while True:
        # Read the latest frame from the webcam
        # if frameNo>0:
        #     oldFrame = frame
        current_time = datetime.now()
        seconds = current_time.second + current_time.microsecond / 1000000

        frame = np.ascontiguousarray(camera.frame)

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        detection_result = detector.detect(image)

        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)


        cv2.imshow('faces', annotated_image)
        key = cv2.waitKey(1)

        plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], fig, ax)

    return

if __name__ == '__main__':
    main()