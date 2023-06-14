import re
import numpy as np
import string
import cv2
class StringUtils:
    def __init__(self):
        pass
    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]','',text)
    def remove_digits(self, text):
        return re.sub(r'\d','',text)
    def remove_extra_spaces(self, text):
        return ''.join(text.split())
    def to_lower(self, text):
        return text.lower()
    def to_upper(self, text):
        return text.upper()
    def normalize(self, text):
        text = self.remove_punctuation(text)
        text = self.remove_extra_spaces(text)
        text = self.to_upper(text)
        return text
    def post_process_text(self, text):
        floor = text.split("_")[0]
        block = text.split("_")[1]
        floor = floor[5:]
        block = block[5:]
        return floor, block
class ImageUtils:
    def __init__(self):
        pass
    def Denoise(self, image):
        """_summary_
        This function denoises the image.
        Args:
            image (_type_): _description_
        """
        return cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    def Resize(self, image, width, height):
        """_summary_
        This function resizes the image.
        Args:
            image (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
        """
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    def Crop(self, image, x, y, width, height):
        """_summary_
        This function crops the image.
        Args:
            image (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
        """
        return image[y:y+height, x:x+width]
    def Rotate(self, image, angle):
        """_summary_
        This function rotates the image.
        Args:
            image (_type_): _description_
            angle (_type_): _description_
        """
        (h, w) = image.shape[:2]
        center = (w/2, h/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    def Translate(self, image, x, y):
        """_summary_
        This function translates the image.
        Args:
            image (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
        """
        M = np.float32([[1, 0, x], [0, 1, y]])
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    def Flip(self, image, flipCode):
        """_summary_
        This function flips the image.
        Args:
            image (_type_): _description_
            flipCode (_type_): _description_
        """
        return cv2.flip(image, flipCode)
    def HistogramEqualization(self, image):
        """_summary_
        This function applies histogram equalization to the image.
        Args:
            image (_type_): _description_
        """
        return cv2.equalizeHist(image)
    def GaussianBlur(self, image, kernelSize):
        """_summary_
        This function applies gaussian blur to the image.
        Args:
            image (_type_): _description_
            kernelSize (_type_): _description_
        """
        return cv2.GaussianBlur(image, (kernelSize, kernelSize), 0)
    def MedianBlur(self, image, kernelSize):
        """_summary_
        This function applies median blur to the image.
        Args:
            image (_type_): _description_
            kernelSize (_type_): _description_
        """
        return cv2.medianBlur(image, kernelSize)
    def BilateralFilter(self, image, diameter, sigmaColor, sigmaSpace):
        """_summary_
        This function applies bilateral filter to the image.
        Args:
            image (_type_): _description_
            diameter (_type_): _description_
            sigmaColor (_type_): _description_
            sigmaSpace (_type_): _description_
        """
        return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
    def Threshold(self, image, threshold, maxValue, thresholdType):
        """_summary_
        This function applies threshold to the image.
        Args:
            image (_type_): _description_
            threshold (_type_): _description_
            maxValue (_type_): _description_
            thresholdType (_type_): _description_
        """
        return cv2.threshold(image, threshold, maxValue, thresholdType)[1]
    def DrawRectangle(self, image, x, y, width, height, color, thickness):
        """_summary_
        This function draws rectangle on the image.
        Args:
            image (_type_): _description_
            x (_type_): _description_
            y (_type_): _description_
            width (_type_): _description_
            height (_type_): _description_
            color (_type_): _description_
            thickness (_type_): _description_
        """
        return cv2.rectangle(image, (x, y), (x+width, y+height), color, thickness)
    def Text_segmentation(self, image):
        """_summary_
        This function segments the text in the image.
        Args:
            image (_type_): _description_
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    def draw_with_alpha(img, draw_function, up, lp, color, width, alpha=0.5):
        overlay = np.zeros_like(img, dtype=np.uint8)
        draw_function(overlay, up, lp, color, width)
        mask = overlay.astype(bool)
        img[mask] = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)[mask]
    def post_processing_image(self, image, xxyy):
        x1, y1, x2, y2 = xxyy
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        h, w, _ = image.shape
        cv2.line(image, (x2, y1), (w, 0), (255, 255, 255), 2)
        cv2.line(image, (x2, y2), (w, y2-y1), (255, 255, 255), 2)
        return image
