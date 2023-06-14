import cv2
import numpy as np
from cvlib.object_detection import ObjectDetection
import pandas as pd
from utils.Tool import StringUtils, ImageUtils
import os
from paddleocr import PaddleOCR


orc = PaddleOCR(use_angle_cls=True, lang="en")

detector = ObjectDetection()
image_utils = ImageUtils()
string_utils = StringUtils()

def take_plates(image, xxyys):
    plates = []
    for xxyy in xxyys:
        x1, y1, x2, y2 = xxyy
        plate = image_utils.Crop(image, x1, y1, x2-x1, y2-y1)

        plates.append(plate)
    return plates

class VehicleSearch:
    def __init__(self):
        pass
    def get_image(self, path = "./image_data"):
        list_image = os.listdir(path=path)
        batch_size = 16
        self.list_image_batch = []
        self.list_name = []
        for i in range(0, len(list_image), batch_size):
            list_batch = list_image[i:i+batch_size]
            list_image_b = []
            for image_path in list_batch:
                image = cv2.imread(os.path.join(path, image_path))
                list_image_b.append(image)
                self.list_name.append(image_path.split(".")[0])
            self.list_image_batch.append(list_image_b)
    def search(self, input_text, load_image = False, confidence_threshold = 0.5, iou_threshold = 0.5):
        """_summary_
        preform search action
        Args:
            input_text (_type_): plate number you want to search
            load_image (bool, optional): reload image or not. Defaults to False.

        Returns:
            original image from cctv: image
            plate image: image
            plate number read from image: string
            bounding box of plate: list
            name of image: string
        """
        if load_image:
            self.get_image()
        for k, batch in enumerate(self.list_image_batch):
            results = detector.multi_predict(batch, conf = confidence_threshold, iou = iou_threshold)
            for i, image in enumerate(batch):
                xxyys = results[i].numpy().boxes.xyxy.astype(int)
                plates = take_plates(image, xxyys)
                for j, plate in enumerate(plates):
                    result = orc.ocr(plate, cls=True)
                    if len(result[0]) == 0:
                        continue
                    text = result[0][0][1][0]
                    text = string_utils.normalize(text)
                    input_text = string_utils.normalize(input_text)
                    if input_text in text:
                        return image, plate, result[0][0][1][0], xxyys[j], self.list_name[k*16+i]
        return None, None, None, None, "No vehicle found" 
    def Multiple_search(self, input_text, load_image = False, confidence_threshold = 0.5, iou_threshold = 0.5):
        """_summary_
        preform multiple search action
        Args:
            input_text (_type_): plate number you want to search
            load_image (bool, optional): reload image or not. Defaults to False.

        Returns: list of
            original image from cctv: image
            plate image: image
            plate number read from image: string
            bounding box of plate: list
            name of image: string
        """
        if load_image:
            self.get_image()
        list_result = []
        for k, batch in enumerate(self.list_image_batch):
            results = detector.multi_predict(batch, conf = confidence_threshold, iou = iou_threshold)
            for i, image in enumerate(batch):
                xxyys = results[i].numpy().boxes.xyxy.astype(int)
                plates = take_plates(image, xxyys)
                for j, plate in enumerate(plates):
                    result = orc.ocr(plate, cls=True)
                    if len(result[0]) == 0:
                        continue
                    text = result[0][0][1][0]
                    text = string_utils.normalize(text)
                    input_text = string_utils.normalize(input_text)
                    if input_text in text:
                        list_result.append((image, plate, result[0][0][1][0], xxyys[j], self.list_name[k*16+i]))
        return list_result


