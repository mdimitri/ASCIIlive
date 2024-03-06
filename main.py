import cv2, time, sys, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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
    resized_image = resized_image ** 0.5
    resized_image *= 255
    resized_image = resized_image.astype(np.uint8)

    # smooth
    # resized_image = cv2.bilateralFilter(resized_image, 5, 5, 5)
    #
    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=0.5) # sharpen
    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=1)  # sharpen
    resized_image = unsharpMask(resized_image, kernelSize=(0, 0), gaussianSigma=5.0) # increase contrast

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


def shiftLines(lines, oldLines, seeds):
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
    if len(seeds) < 5*len(lines[0]): # at least a few per column
        for i in np.arange(0, np.random.randint(10, width)):
            seeds.append([np.random.randint(0, width), np.random.randint(0, height)])

    # reduce row number of seeds quickly
    whichones = np.random.randint(0, 10, len(seeds))
    # if np.mod(milliseconds, 10) <= 10:
    for idx, seed in enumerate(seeds):
        if whichones[idx] < 7:
            seed[1] = seed[1] + 1

    # remove finished seeds
    seeds = [seed for seed in seeds if seed[1] < height]

    return oldLines, seeds

def main():
    cv2.namedWindow('ASCII', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('ASCII', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # this is the magic!
    targetResolution = [1280//2, 800//2]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, targetResolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, targetResolution[1])
    cap.set(cv2.CAP_PROP_FPS, 60)

    scaleFactor = 1
    output_width = int(190 / scaleFactor)
    frameNo = 0
    seeds = []

    # Define the ASCII characters to represent different intensity levels
    # ascii_chars = "@%#*+=-:. "[::-1]
    # ascii_chars = ['@', '8', '#', 'B', 'W', 'O', '0', 'Q', 'L', 'C', 'J', 'P', 'X', 'Z', 'U', 'Y', 'Z', 'm', 'w', 'o', 'a', 'h', 'k', 'b', 'd', 'p', 'q', 'u', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', 'v', 'u', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', 'n', 'r', 'j', 'v', 'y', 'c', 'x', 'z', '!', 'i', 'l', 'I', ';', ':', ',', '.', '`', ' ']
    ascii_chars =  "`.-:_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    # ascii_chars = r'$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`\'.'[::-1]
    # ascii_chars = "@MBHENR#KWXDFPQASUZbdehx*8Gm&04LOVYkpq5Tagns69owz$CIu23Jcfry%1v7l+it[] {}?j|()=~!-/<>\"^_';,:`. "[::-1]
    # ascii_chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "[::-1]
    rasterDict = {}
    for char in ascii_chars:
        rasterDict[char] = [np.zeros(0), []] # the raster image and it's size

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if frame.shape[0] < targetResolution[1]:
            cutWidth = int((frame.shape[1] - frame.shape[0] * targetResolution[0] / targetResolution[1]) / 2)
            frame = frame[:, cutWidth : -cutWidth, :]
        frame = np.flip(frame, axis=1)

        ascii_art, new_height, lines, resized_image = image_to_ascii(frame, ascii_chars=ascii_chars, output_width=output_width)

        if frameNo>0:
            # shift lines matrix-style
            oldLines, seeds = shiftLines(lines, oldLines=oldLines, seeds=seeds)
        else:
            oldLines = lines

        charHeight = int(10 * scaleFactor)
        charWidth = int(10 * scaleFactor)
        # Create a blank canvas to render ASCII art using OpenCV
        canvas_width, canvas_height = output_width * charWidth, new_height * charHeight
        if frameNo == 0:
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) # initialize a blank canvas

        canvasHighlight = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # initialize a blank canvas

        # boost colors, and rotate hue over time
        current_time = datetime.now()
        seconds = current_time.second + current_time.microsecond / 1000000

        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2HSV)
        resized_image[:, :, 0] = np.mod(resized_image[:, :, 0] + seconds / (60 / (180)), 180)
        resized_image[:, :, 1] = ((resized_image[:, :, 1] / 255.0) ** 0.25) * 255
        resized_image[:, :, 2] = ((resized_image[:, :, 2] / 255.0) ** 1) * 255
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_HSV2RGB)

        resized_image_viz = cv2.resize(resized_image,(canvas_width, canvas_height), interpolation=cv2.INTER_AREA)

        # Render ASCII characters on the canvas
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = np.minimum(charHeight, charWidth) / 25
        font_thickness = 1


        # go over each seed point and update the canvas
        for seed in seeds:
            char = oldLines[seed[1]][seed[0]]
            # check if we have pre-computed raster for this character, if not compute it and store it in a dict
            if not (rasterDict[char][1]):
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
                char_image_resized = cv2.resize(char_image, (charWidth, charHeight), cv2.INTER_NEAREST)
                # store it for later use
                rasterDict[char][0] = char_image_resized
                rasterDict[char][1] = textSize
            else:
                # load it from the dict
                char_image_resized = rasterDict[char][0]
                textSize = rasterDict[char][1]

            # paste it into the canvas
            canvas[seed[1]*charHeight:seed[1]*charHeight + charHeight,
                    seed[0]*charWidth:seed[0]*charWidth + charWidth] = char_image_resized
            canvasHighlight[seed[1] * charHeight:seed[1] * charHeight + charHeight,
                seed[0] * charWidth:seed[0] * charWidth + charWidth] = ((char_image_resized / 255.0) ** 0.6) * 255

        # for idxr, line in enumerate(oldLines):
        #     x_position = 0  # Starting position for each line
        #     for idxc, char in enumerate(line):
        #         char_image = np.zeros((charHeight, charWidth, 3), dtype=np.uint8) # rectangle for each character
        #         # get the size of the rasterized letter
        #         textSize = cv2.getTextSize(char, fontFace=font, fontScale=font_scale, thickness=font_thickness)[0]
        #         # put it in the center of the tile
        #         color = (255, 255, 255)
        #         # color = resized_image[idxr, idxc, :]
        #         cv2.putText(char_image, char, (charWidth // 2 - textSize[0] // 2,
        #                                        charHeight // 2 + textSize[1] // 2),
        #                     font, font_scale, (int(color[0]), int(color[1]), int(color[2])), font_thickness, cv2.LINE_AA)
        #
        #         # Resize the character image to fit the canvas size
        #         char_image_resized = cv2.resize(char_image, (charWidth, charHeight), cv2.INTER_NEAREST)
        #
        #         canvas[y_position:y_position + charHeight, x_position:x_position + charWidth] = char_image_resized[:, :charWidth, :]
        #         x_position += charWidth  # Move to the next position for the next character
        #     y_position += charHeight  # Move to the next line

        canvasBlend = np.where(canvasHighlight!=(0,0,0), canvasHighlight, canvas)
        cv2.imshow('ASCII',  cv2.resize(canvasBlend, dsize=(1920, 1200), interpolation=cv2.INTER_NEAREST))
        # cv2.imshow('ASCII', np.hstack((resized_image_viz, canvasBlend)))
        if cv2.waitKey(1) == ord('x'):
            return

        frameNo += 1

    return

if __name__ == '__main__':
    main()
