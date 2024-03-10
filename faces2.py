import cv2, time, sys, os, string
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread
import mediapipe as mp
from funnyNames import funny_names

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def savePic(pic):
    path = './Wall of fame'
    current_datetime = datetime.now()
    pathPic = os.path.join(path, current_datetime.strftime("%Y%m%d%H%M%S") + '.jpg')
    cv2.imwrite(pathPic, pic, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.waitKey(1000)
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

    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
    #     landmark_drawing_spec=None,
    #     connection_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=0, color=(64, 64, 64)))
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=0, color=(255, 255, 255)))
    # solutions.drawing_utils.draw_landmarks(
    #     image=annotated_image,
    #     landmark_list=face_landmarks_proto,
    #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
    #       landmark_drawing_spec=None,
    #       connection_drawing_spec=solutions.drawing_utils.DrawingSpec(thickness=4, circle_radius=0, color=(255, 255, 255)))

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
    def __init__(self, targetResolution, targetSize):
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, targetResolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, targetResolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.targetResolution = targetResolution
        self.targetSize = targetSize
        # self.cropSize = cropSize

        # read in one frame
        self.frame = self.cap.read()[1]
        # if self.frame.shape[0] < self.targetResolution[1]:
        #     cutWidth = int((self.frame.shape[1] - self.frame.shape[0] * self.targetResolution[0] / self.targetResolution[1]) / 2)
        #     self.frame = self.frame[:, cutWidth: -cutWidth, :]
        self.frame = np.flip(self.frame, axis=1)
        # crop central portion of grabbed frame
        # self.frame = cropFrame(self.frame, crop_dimension=self.cropSize)
        # start async
        self.start()


    def run(self):
        while True:
            self.frame = self.cap.read()[1]
            # if self.frame.shape[0] < self.targetResolution[1]:
            #     cutWidth = int((self.frame.shape[1] - self.frame.shape[0] * self.targetResolution[0] / self.targetResolution[1]) / 2)
            #     self.frame = self.frame[:, cutWidth: -cutWidth, :]
            self.frame = np.flip(self.frame, axis=1)
            # crop central portion of grabbed frame
            # self.frame = cropFrame(self.frame, crop_dimension=self.cropSize)

def rotate_image(image, angle):
    # Get the center of the image
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the new size of the rotated image
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    new_width = int(height * sin_theta + width * cos_theta)
    new_height = int(height * cos_theta + width * sin_theta)

    # Adjust the rotation matrix for the new size
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), flags=cv2.INTER_LINEAR)

    return rotated_image

def draw_smile_meter(image, smileMeter):
    width = image.shape[1]
    height = image.shape[0]
    top = height*0.95 - smileMeter * (height * 0.9)
    rect_start = (int(width * 0.88), int(top))  # top left
    rect_end = (int(width * 0.93), int(height * 0.95))  # bottom right
    color = (255, 255, 255)
    thickness = 5
    cv2.rectangle(image, rect_start, rect_end, color, thickness)
    color = cv2.applyColorMap(np.array([smileMeter*60], dtype=np.uint8), cv2.COLORMAP_HSV).ravel()
    color = (int(color[0]), int(color[1]), int(color[2]))
    cv2.rectangle(image, rect_start, rect_end, color, -1)
    cv2.putText(image, 'Smile-o-Meter', org=(width - 175, height - 8),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7,thickness=1,
                color=(255, 255, 255), lineType=cv2.LINE_4)

    return image


def draw_names(image, detection_result, funnyNames, seconds):

    height = image.shape[0]; width = image.shape[1]
    if len(detection_result.face_blendshapes):
        np.random.seed(int((seconds/10)) * 10)
        for face_landmarks in detection_result.face_landmarks:
            # compute coordinate center
            x = 0; y = 0; minx = 1; miny = 1;
            for landmark in face_landmarks:
                x += landmark.x
                y += landmark.y
                if landmark.x<minx:
                    minx=landmark.x
                if landmark.y<miny:
                    miny=landmark.y
            x /= len(face_landmarks)
            y /= len(face_landmarks)

            # pick random name based on 10 second seed cycle
            name = np.random.choice(funnyNames)

            cv2.putText(image, name, org=(int(minx * width), int((miny-0.02) * height)),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.9, thickness=1,
                        color=(255, 255, 255), lineType=cv2.LINE_4)

    return image
def main():
    # setting for the webcam
    targetResolution = [1920 // 1, 1080 // 1]
    targetSize = (1920 // 2, 1080 // 2)
    detectionSubsample = 1


    camera = RGBcamera(targetResolution=targetResolution, targetSize=targetSize)
    cv2.waitKey(100)

    base_options = python.BaseOptions(model_asset_path='./weights/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=2)
    detector = vision.FaceLandmarker.create_from_options(options)

    fig, ax = plt.subplots(figsize=(12, 12))
    amplitude = 5
    phaseFact = 10
    hsvFact = 1
    globalPhase = 0
    hsvPhase = 0

    rotation = 0
    smileMeter = 0

    while True:
        # Read the latest frame from the webcam
        # if frameNo>0:
        #     oldFrame = frame
        current_time = datetime.now()
        seconds = current_time.minute * 60 + current_time.second + current_time.microsecond / 1000000

        frame = cv2.resize(camera.frame, targetSize, interpolation=cv2.INTER_NEAREST)
        background = np.copy(frame)


        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.resize(frame, (0, 0), fx = 1/detectionSubsample, fy = 1/detectionSubsample, interpolation=cv2.INTER_NEAREST))

        detection_result = detector.detect(image)

        smileMeter = np.clip(smileMeter - 0.02, 0, 1)

        if len(detection_result.face_blendshapes):
            browInnerUp = 0
            mouthPucker = 0
            mouthSmileLeft = 0
            mouthSmileRight = 0
            eyeLookInLeft = 0
            eyeLookInRight = 0
            eyeLookOutLeft = 0
            eyeLookOutRight = 0
            faceCount = len(detection_result.face_blendshapes)
            # control the wiggle phase with facial expressions
            for face_blendshapes in detection_result.face_blendshapes:
                face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
                face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]

                browInnerUp += face_blendshapes_scores[3]
                mouthPucker += face_blendshapes_scores[38]

                mouthSmileLeft += face_blendshapes_scores[44]
                mouthSmileRight += face_blendshapes_scores[45]

                eyeLookInLeft += face_blendshapes_scores[13]
                eyeLookInRight += face_blendshapes_scores[14]
                eyeLookOutLeft += face_blendshapes_scores[15]
                eyeLookOutRight += face_blendshapes_scores[16]

            browInnerUp /= faceCount
            mouthPucker /= faceCount
            mouthSmileLeft /= faceCount
            mouthSmileRight /= faceCount
            eyeLookInLeft /= faceCount
            eyeLookInRight /= faceCount
            eyeLookOutLeft /= faceCount
            eyeLookOutRight /= faceCount

            learnFact = 0.05
            smileMeter += (learnFact * (mouthSmileLeft + mouthSmileRight)/2)
            smileMeter = np.clip(smileMeter, 0, 1)

            learnFact = 0.9
            amplitude = amplitude * learnFact + (5 + 5 * (mouthSmileLeft + mouthSmileRight)/2) * (1 - learnFact)

            learnFact = 0.9
            phaseFact = phaseFact * learnFact + (10 + 50 * browInnerUp) * (1 - learnFact)

            learnFact = 0.9
            hsvFact = hsvFact * learnFact + (1 + 15 * mouthPucker) * (1 - learnFact)

            learnFact = 1.0
            rotation -= learnFact * (eyeLookInLeft - eyeLookOutLeft)

        annotated_image = draw_landmarks_on_image(0*image.numpy_view(), detection_result)

        annotated_image = draw_names(annotated_image, detection_result, funny_names, seconds)

        annotated_image = cv2.resize(annotated_image, (0, 0), fx = detectionSubsample, fy = detectionSubsample, interpolation=cv2.INTER_NEAREST)

        # draw smile-o-meter
        annotated_image = draw_smile_meter(annotated_image, smileMeter)

        # make a trippy background

        background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        hsvPhase += hsvFact * seconds/20000
        # boost colors, and rotate hue over time
        background[:, :, 0] = np.mod(background[:, :, 0] + hsvPhase / (10 / (180)), 180)
        background[:, :, 1] = background[:, :, 1] + np.minimum(128, 255-background[:, :, 1])
        background[:, :, 2] = background[:, :, 2] + np.minimum(64, 255-background[:, :, 2])
        background = cv2.cvtColor(background, cv2.COLOR_HSV2BGR)


        globalPhase += phaseFact * ((seconds / 3600.0) / 10)
        background = apply_wiggly_pattern(background, frequency=5, amplitude=amplitude, phase=globalPhase)


        canvasBlend = np.where(np.repeat((np.sum(annotated_image, axis=2) == 0)[:, :, np.newaxis], 3, axis=2), background, annotated_image)

        # canvasBlend = rotate_image(canvasBlend, rotation)
        cv2.imshow('faces', canvasBlend)
        cv2.waitKey(1)

        if smileMeter == 1:
            # store both photos
            savePic(cv2.cvtColor(canvasBlend, cv2.COLOR_BGR2RGB))
            savePic(frame)
            smileMeter = 0.0


        # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0], fig, ax)

    return

if __name__ == '__main__':
    main()