import cv2
import numpy as np
import os
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path="./YOLOV8N-LicensePlate2/train/weights/best.pt", device = None):
        print("Dectector model is loading...")
        self.model = YOLO(model_path)
        self.device = device
        print("Dectector model is ready now")
    def predict(self, frame, conf = 0.4, iou = 0.5, plot = False):
        """_summary_
        predict the frame and return the result
        Args
            frame (_type_): frame to predict numpy array
            specific_class (list optional): List of class you want to extract. Defaults to None. Mean that you want to extract all class.

        Returns:
            list: list of result object with class, confidence, and bounding box
        """
        results = self.model.predict(frame, conf=conf, iou=iou, device=self.device, verbose=False)
        b_results = results[0].numpy().boxes
        xxyys = b_results.xyxy.astype(int)
        if plot:
            return xxyys, results[0].plot()
        else:
            return xxyys
    def multi_predict(self, frames, conf = 0.4, iou = 0.5):
        results = self.model.predict(frames, conf=conf, iou=iou, device=self.device, verbose=False)
        return results

