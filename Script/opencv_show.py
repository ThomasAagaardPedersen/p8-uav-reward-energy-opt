# DRL Oprimized Drone Path
# Created by: AL, TAP, SFMS // APEL1-1
# P7 Project
# Aalborg Univeristy Esbjerg AAU / Fall_2022

# In settings.json first activate computer vision mode:
# https://github.com/Microsoft/AirSim/blob/main/docs/image_apis.md#computer-vision-mode

from re import I
import setup_path
import airsim

# requires Python 3.5.3 :: Anaconda 4.4.0
# pip install opencv-python
import cv2
import os
import time
import sys
import numpy as np
import torch
import threading
import pandas as pd

from PIL import Image
from pathlib import Path

model_path = Path(__file__).parent / "path/to/best_WTB.pt"

model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=model_path, force_reload=True
)  #  local model
print("Model has been downloaded and created")


def detectAndMark(image):
    result = model(image)
    result.print()
    objs = result.pandas().xyxy[0]
    objs_name = objs.loc[objs["name"] == "WTB"]
    height = image.shape[0]
    width = image.shape[1]
    x_middle = 0
    y_middle = 0
    x_min = None
    y_min = None
    x_max = None
    y_max = None
    confidence = 0
    try:
        obj = objs_name.iloc[0]
        x_min = obj.xmin
        y_min = obj.ymin
        x_max = obj.xmax
        y_max = obj.ymax
        confidence = obj.confidence
        x_middle = x_min + (x_max - x_min) / 2
        y_middle = y_min + (y_max - y_min) / 2

        print(objs)

        x_middle = round(x_middle, 0)
        y_middle = round(y_middle, 0)
        # Calculate the distance from the middle of the camera frame view, to the middle of the object
        x_distance = x_middle - width / 2
        y_distance = y_middle - height / 2

        cv2.rectangle(
            image,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        cv2.circle(image, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
        cv2.circle(image, (int(width / 2), int(height / 2)), 5, (0, 0, 255), 2)
        cv2.line(
            image,
            (int(x_middle), int(y_middle)),
            (int(width / 2), int(height / 2)),
            (0, 0, 255),
            2,
        )
        cv2.rectangle(
            image,
            (int((width / 2) - 200),(int((height / 2) - 200))),
            (int((width / 2) + 200),(int((height / 2) + 200))),
            (255, 0, 0),
            2,
        )
    except:
        print("Error")
        print(objs)
        data = pd.DataFrame({tuple(confidenceData)})
        data.to_excel("confidenceData.xlsx", sheet_name="sheet1", index=False)
    return image, x_min, y_min, x_max, y_max, confidence


cameraTypeMap = {
    "depth": airsim.ImageType.DepthVis,
    "segmentation": airsim.ImageType.Segmentation,
    "seg": airsim.ImageType.Segmentation,
    "scene": airsim.ImageType.Scene,
    "disparity": airsim.ImageType.DisparityNormalized,
    "normals": airsim.ImageType.SurfaceNormals,
}


client = airsim.MultirotorClient()

print("Connected: now while this script is running, you can open another")
print("console and run a script that flies the drone and this script will")
print("show the depth view while the drone is flying.")

help = False

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

prev_x_min = 0
prev_y_min = 0
prev_x_max = 0
prev_y_max = 0
confidenceData = []

while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    # rawImage = client.simGetImage("0", cameraTypeMap[cameraType])
    rawImages = client.simGetImages(
        [airsim.ImageRequest("high_res", airsim.ImageType.Scene, False, False)]
    )
    # rawImageDepth = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)])

    # rawImageDepth = client.simGetImage("high_res", airsim.ImageType.DepthVis)
    # rawImageDepth = client.simGetImage("high_res", airsim.ImageType.DepthPerspective, True)
    # responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)])
    # responses = client.simGetImages(
    #     [airsim.ImageRequest("high_res", airsim.ImageType.DepthPerspective, True)]
    # )

    if rawImages == None:
        print("Camera is not returning image, please check airsim for error messages")
        sys.exit(0)
    else:
        # High resolution, color image
        response = rawImages[0]
        rawImage = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        rawImage = rawImage.reshape(response.height, response.width, 3)
        rawImage, x_min, y_min, x_max, y_max, confidence = detectAndMark(rawImage)
        cv2.imshow("FPV", rawImage)

        # confidenceData.append(confidence)
        # data = pd.DataFrame({tuple(confidenceData)})
        # data.to_excel("confidenceData.xlsx", sheet_name="sheet1", index=False)

        if x_min == None or y_min == None or x_max == None or y_max == None:
            x_min = prev_x_min
            y_min = prev_y_min
            x_max = prev_x_max
            y_max = prev_y_max
        else:
            prev_x_min = x_min
            prev_y_min = y_min
            prev_x_max = x_max
            prev_y_max = y_max

        # Depth camera

        # img_depth = np.asarray(responses[0].image_data_float)
        # img_depth = img_depth.reshape(responses[0].height, responses[0].width)
        # print("Depth max:", np.nanmax(img_depth))
        # img_depth[img_depth > 16000] = np.nan

        # print("test shape original: ", img_depth.shape)
        # img_depth = cv2.resize(img_depth, (1920, 1080), interpolation=cv2.INTER_AREA)
        # print("test shape interpolated: ", img_depth.shape)

        # img_depth = img_depth[int(y_min) : int(y_max), int(x_min) : int(x_max)]
        # print("test shape cut: ", img_depth.shape)

        # x_small_val = 16000
        # y_small_val = 16000
        # x_small = 0
        # y_small = 0
        # print("Dimensions: ", img_depth.shape[0], " ", img_depth.shape[1])
        
        # GOOD STUFF
        # cv2.imshow("FPV", img_depth)
        # cv2.imshow("depth", depth_map)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord("q") or key == ord("x"):
        break
