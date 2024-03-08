import cv2, time, sys, os, string
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread

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

def unsharpMask(image, kernelSize=(0,0), gaussianSigma=2):
    gaussian_3 = cv2.GaussianBlur(image, kernelSize, gaussianSigma)
    unsharp_image = cv2.addWeighted(image, 2.0, gaussian_3, -1.0, 1)
    return unsharp_image
def image_to_ascii(image, ascii_chars, output_width=100):
    # Resize the image to fit the desired output width
    aspect_ratio = image.shape[0] / image.shape[1]
    new_height = int(np.ceil(output_width * aspect_ratio))
    resized_image = cv2.resize(image, (output_width, new_height), interpolation=cv2.INTER_NEAREST)

    resized_image = resized_image / 255.0
    resized_image = resized_image ** 0.3
    resized_image *= 255
    resized_image = resized_image.astype(np.uint8)

    # resized_image = resized_image + ((255 - resized_image) >> 1)

    # smooth
    # resized_image = cv2.bilateralFilter(resized_image, 5, 5, 5)

    # # apply histogram stretching
    # for c in np.arange(0, 3):
    #     resized_image[:,:,c] = histStretch(resized_image[:,:,c])

    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=1)   # sharpen
    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=2)   # sharpen
    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=5) # increase contrast


    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)



    # Convert each pixel to ASCII character based on intensity
    ascii_image = ''
    grayscale_image = Image.fromarray(np.uint8(grayscale_image))


    jitterRange = 1
    jitter = np.random.randint(-jitterRange, jitterRange, size=np.prod(grayscale_image.size))
    for idx, pixel_value in enumerate(grayscale_image.getdata()):
        ascii_image += ascii_chars[jitterRange + int(pixel_value // (256 / (len(ascii_chars) - 2 * jitterRange))) + jitter[idx]]

    # for idx, pixel_value in enumerate(grayscale_image.getdata()):
    #     ascii_image += ascii_chars[int(pixel_value // (256 / len(ascii_chars)))]

    # Split the ASCII image into lines
    lines = [ascii_image[i:i + output_width] for i in range(0, len(ascii_image), output_width)]
    ascii_art = '\n'.join(lines)

    return ascii_art, new_height, lines, resized_image


def savePic(pic):
    path = './Wall of fame'
    current_datetime = datetime.now()
    pathPic = os.path.join(path, current_datetime.strftime("%Y%m%d%H%M%S") + '.jpg')
    cv2.imwrite(pathPic, pic, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return
def shiftLines(lines, oldLines, seeds, frameDiff):
    # keep the old lines, except at seed positions where data is overwritten with new lines
    # replace new line characters into old lines
    ENC_TYPE = "ascii"
    for seed in seeds:
        s = bytearray(oldLines[seed[1]], ENC_TYPE)
        s[seed[0]] = ord(lines[seed[1]][seed[0]])
        oldLines[seed[1]] = s.decode(ENC_TYPE)

    width = len(lines[0])
    height = len(lines)

    # # every second add a few seed points
    # current_time = datetime.now()
    # milliseconds = current_time.microsecond // 1000
    # seconds = current_time.second + current_time.microsecond / 1000000

    # if np.mod(milliseconds, 1000) < 1000:
    # while len(seeds) < 5*len(lines[0]):
    #     seeds.append([np.random.randint(0, width), np.random.randint(0, height)])

    # # probabilistic seeding near frame difference regions
    # frameDiff = cv2.resize(frameDiff, (width, height), cv2.INTER_NEAREST)
    # frameDiff = frameDiff ** 3.0
    # flat_frame_diff = frameDiff.flatten()
    # probabilities = flat_frame_diff / np.sum(flat_frame_diff)
    # random_indices = np.random.choice(np.arange(len(probabilities)), size= 10*len(lines[0]) - len(seeds), p=probabilities)
    # row_indices, col_indices = np.unravel_index(random_indices, frameDiff.shape)
    # row_indices = np.clip(row_indices - height//20, 0, height)
    # seeds += [list(i) for i in zip(col_indices, row_indices)]

    # if the number of seeds is lower than the desired amount, spawn new ones
    while len(seeds) < 15*len(lines[0]):
        highlight = np.random.randint(0, 10) # is this seed point going to be highlighted in the image
        seeds.append([np.random.randint(0, width), np.random.randint(2, height//4), highlight > 5]) # get random coordinates and add to list
        seeds.append([seeds[-1][0], seeds[-1][1] - 1, 0]) # add another one, one row above
        seeds.append([seeds[-1][0], seeds[-1][1] - 1, 0]) # and another one

    # if len(seeds) < 1*len(lines[0]): # at least a few per column
    #     for i in np.arange(0, np.random.randint(10, width)):
    #         seeds.append([np.random.randint(0, width), np.random.randint(0, height//2)])

    # update positions, fall down
    whichones = np.random.randint(0, 10, len(seeds))
    # if np.mod(milliseconds, 10) <= 10:

    # move three seeds at a time
    for i in np.arange(0, len(seeds)-3, 3):
        if whichones[i] > 1:
            seeds[i][1]   = seeds[i][1] + 3
            seeds[i+1][1] = seeds[i+1][1] + 3
            seeds[i+2][1] = seeds[i+2][1] + 3

    # for idx, seed in enumerate(seeds):
    #     # if whichones[idx] < 10:
    #     seed[1] = seed[1] + 3 # move each seed point 3 steps below, since we spawn 3 seed points in series, they will move seamlessly


    # remove finished seeds
    # seeds = [seed for seed in seeds if ((seed[1] < (height)) and (seed[1] >= 0))]

    keep = np.ones(len(seeds),dtype=np.int8)
    for idx, seed in enumerate(seeds):
        if keep[idx] and not(seed[1] < height and seed[1] >=0):
            # out of bounds, mark this and the previous 2 for deletion
            keep[idx:idx+3] = 0

    seeds = [seed for i, seed in enumerate(seeds) if keep[i]]

    return oldLines, seeds

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

def histStretch(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    img2 = cdf[img]
    return img2

def sortCharsVisually(charHeight, charWidth, font, font_scale, font_thickness):
    printable_ascii_characters = list(string.printable)
    # remove carriage returns and tabs and whitespaces
    printable_ascii_characters = [char for char in printable_ascii_characters if not(char in ['\x0a', '\x0b', '\x0c', '\r', '\t', ' '])]

    charWeight = np.zeros(len(printable_ascii_characters))
    for idx, char in enumerate(printable_ascii_characters):
        char_image = np.zeros((charHeight, charWidth, 3), dtype=np.uint8)  # rectangle for each character
        # get the size of the rasterized letter
        textSize = cv2.getTextSize(char, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
        # put it in the center of the tile
        color = (255, 255, 255)
        cv2.putText(char_image, char, (charWidth // 2 - textSize[0] // 2,
                                       charHeight // 2 + textSize[1] // 2),
                    font, font_scale, (int(color[0]), int(color[1]), int(color[2])), font_thickness, cv2.LINE_AA)
        charWeight[idx] = (char_image / 255.0).sum() / np.prod(char_image.shape)

    idx = np.argsort(charWeight)
    printable_ascii_characters = [printable_ascii_characters[i] for i in idx]
    return printable_ascii_characters, charWeight
def main():
    cv2.namedWindow('ASCII', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('ASCII', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # setting for the webcam
    targetResolution = [1280 // 2, 800 // 2]
    # crop the central portion
    cropSize = targetResolution[0] // 2

    camera = RGBcamera(targetResolution=targetResolution, cropSize=cropSize)
    cv2.waitKey(100)


    scaleFactor = 1.25
    output_width = int(192 / scaleFactor)

    charHeight = int(10 * scaleFactor)
    charWidth = int(10 * scaleFactor)

    # Render ASCII characters on the canvas
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = np.minimum(charHeight, charWidth) / 25
    font_thickness = 1

    # precompute the brightness levels of all printable characters
    ascii_chars, charWeight = sortCharsVisually(charHeight, charWidth, font, font_scale, font_thickness)

    frameNo = 0
    seeds = []

    # Define the ASCII characters to represent different intensity levels
    # ascii_chars = "@%#*+=-:. "[::-1]
    # ascii_chars = ['@', '8', '#', 'B', 'W', 'O', '0', 'Q', 'L', 'C', 'J', 'P', 'X', 'Z', 'U', 'Y', 'Z', 'm', 'w', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'u', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', 'v', 'u', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', '!', 'i', 'l', 'I', ';', ':', ',', '.', '`', ' ']
    # ascii_chars =  "`.-:_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    # ascii_chars = r'$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`\'.'[::-1]
    # ascii_chars = "@MBHENR#KWXD8GFPQASUZ&04LOVY5Tbdehx*mkpqagns69owz$CIu23Jcfry%1v7l+it[] {}?j|()=~!-/<>\"^_';,:`. "[::-1]
    # ascii_chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]


    rasterDict = {}
    for char in ascii_chars:
        rasterDict[char] = [np.zeros(0), []] # the raster image and it's size

    while True:
        # Read the latest frame from the webcam
        # if frameNo>0:
        #     oldFrame = frame

        frame = camera.frame

        # if frameNo>0:
        #     frameDiff = np.clip(np.mean(np.abs(
        #         cv2.GaussianBlur(frame, (11, 11), 0)-
        #         cv2.GaussianBlur(oldFrame, (11, 11), 0)), axis=2) / 255.0, 0, 1)

        ascii_art, new_height, lines, resized_image = image_to_ascii(frame, ascii_chars=ascii_chars, output_width=output_width)

        if frameNo>0:
            # shift lines matrix-style
            oldLines, seeds = shiftLines(lines, oldLines=oldLines, seeds=seeds, frameDiff=[])
        else:
            oldLines = lines

        # Create a blank canvas to render ASCII art using OpenCV
        canvas_width, canvas_height = output_width * charWidth, new_height * charHeight
        if frameNo == 0:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) # initialize a blank canvas

        canvasHighlight = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # initialize a blank canvas

        # boost colors, and rotate hue over time
        current_time = datetime.now()
        seconds = current_time.second + current_time.microsecond / 1000000

        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
        resized_image[:, :, 0] = np.mod(resized_image[:, :, 0] + seconds / (60 / (180)), 180)
        resized_image[:, :, 1] = ((resized_image[:, :, 1] / 255.0) ** 0.5) * 255
        resized_image[:, :, 2] = ((resized_image[:, :, 2] / 255.0) ** 2) * 255
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_HSV2BGR)

        # resized_image_viz = cv2.resize(resized_image,(canvas_width, canvas_height), interpolation=cv2.INTER_NEAREST)

        # darken canvas
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
        canvas[:, :, 2] = np.clip(canvas[:, :, 2].astype(np.int16) - 2, 0, 255).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_HSV2BGR)


        # go over each seed point and update the canvas
        for seed in seeds:
            char = oldLines[seed[1]][seed[0]]
            char_image = np.zeros((charHeight, charWidth, 3), dtype=np.uint8)  # rectangle for each character
            # get the size of the rasterized letter
            textSize = cv2.getTextSize(char, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
            # put it in the center of the tile
            # color = (255, 255, 255)
            color = resized_image[seed[1], seed[0], :]
            cv2.putText(char_image, char, (charWidth // 2 - textSize[0] // 2,
                                           charHeight // 2 + textSize[1] // 2),
                        font, font_scale, (int(color[0]), int(color[1]), int(color[2])), font_thickness, cv2.LINE_AA)

            # Resize the character image to fit the canvas size
            # char_image_resized = cv2.resize(char_image, (charWidth, charHeight), cv2.INTER_NEAREST )

            # # store it for later use
            # rasterDict[char][0] = char_image_resized
            # rasterDict[char][1] = textSize

            # paste it into the canvas
            canvas[seed[1]*charHeight:seed[1]*charHeight + charHeight,
                    seed[0]*charWidth:seed[0]*charWidth + charWidth] = char_image#_resized
            highlightAmount = (0.5 if seed[2] == 1 else 1.0)
            canvasHighlight[seed[1] * charHeight:seed[1] * charHeight + charHeight,
                seed[0] * charWidth:seed[0] * charWidth + charWidth] = ((char_image / 255.0) ** highlightAmount) * 255


        canvasBlend = np.where(canvasHighlight!=(0,0,0), canvasHighlight, canvas)
        cv2.imshow('ASCII',  cv2.resize(canvasBlend, dsize=(1920, 1200), interpolation=cv2.INTER_NEAREST))

        # if frameNo>0:
            # cv2.imshow('ASCII', np.hstack((canvasBlend, canvasBlend))
            # cv2.imshow('ASCII', frameDiff)
        key = cv2.waitKey(1)
        if key == ord('x'):
            return
        elif key == ord(' '):
            savePic(canvasBlend)


        frameNo += 1

    return

if __name__ == '__main__':
    main()
