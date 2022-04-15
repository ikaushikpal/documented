import io
import numpy as np
from typing import List
from img2pdf import convert
import cv2


class PDF:
    '''
    It takes a list of images, 
    and generates a pdf with all the given images.
    '''

    def __init__(self, images: List[np.ndarray]) -> None:
        '''
        Constructor for the class.

        @Parameters:
            images: List[np.ndarray]: List of numpy arrays. 
    
        @Returns:
            None
        '''

        if type(images) not in (tuple, list, np.ndarray):
            raise TypeError("images must be a list of numpy arrays")

        if type(images) == np.ndarray:
            images = [images]
        
        # generating the list of images in binary format.
        self.__formImage(images)

    def __numpyToBinary(self, image: np.ndarray) -> bytes:
        '''
        This method converts all images into binary format.

        @Parameters:
            image: np.ndarray: image in numpy array format.
        
        @Returns:
            bytes: binary image in bytes format.
        '''

        # generate image from buffer
        is_success, buffer = cv2.imencode(".jpg", image)

        # check if image is successfully converted to binary format
        # if not, raise exception
        if not is_success:
            raise Exception("Could not convert image to binary")

        # convert direct raw memory to binary object
        io_buf = io.BytesIO(buffer)

        # reading whole binary object
        return io_buf.read()

    def __formImage(self, images: List[np.ndarray]) -> None:
        '''
        Generate array of images in binary format.

        @Brief:
            This method converts one image into binary format, then append to the list. \n
            This process continues until all images are converted to binary format.

        @Parameters:
            images: List[np.ndarray]: List of numpy arrays.
        
        @Returns:
            None
        '''

        self.images = []

        for image in images:
            binaryImage = self.__numpyToBinary(image)
            self.images.append(binaryImage)
    
    def convert(self) -> bytes:
        '''
        converts the array of binary into a pdf format. 

        @Brief:
            By calling convert method from img2pdf, it converts the binary array into a pdf format. \n
            img2pdf.convert method takes a list of binary images as input.

        @Parameters:
            None
        
        @Returns:
            bytes: binary pdf file.
        '''

        return convert(self.images)
