import numpy as np
import cv2


class GenerateDocument:
    '''
    This class scans for a document portion of the given image,
    and returns a image which is most suitbly a documents found from the image.
    '''

    MINIMUM_CONTOUR_AREA = 5000
    IMAGE_WIDTH = 540
    IMAGE_HEIGHT = 640

    def preProcessing(self, image: np.ndarray) -> np.ndarray:
        '''
        @brief
            1. Here we convert the color of an image from BGR space to Grayscale first
            2. Then we blur the image to reducing noise
            3. Then we detect the edges in an image using Canny's edge detection method
            4. Then we perform erosion on the image and return it

        @parameters
            image: numpy.ndarray
        
        @return
            image: numpy.ndarray a image with modified numpy 
        '''
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageBlur = cv2.GaussianBlur(imageGray, (5, 5), 1)
        imageCanny = cv2.Canny(imageBlur, 200, 200)

        kernel = np.ones((5, 5))
        imageDial = cv2.dilate(imageCanny, kernel, iterations=2)
        imageThreshold = cv2.erode(imageDial, kernel, iterations=1)

        return imageThreshold

    def getBiggestContour(self, image: np.ndarray) -> np.ndarray:
        '''
        Here we get the biggest contour of length 4, 
        
        @parameters
            image: numpy.ndarray

        @return
            image: numpy.ndarray
        '''
        biggestContour, maxContourArea = np.array([]), 0
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contourArea = cv2.contourArea(contour)

            if contourArea > GenerateDocument.MINIMUM_CONTOUR_AREA:
                contourPerimeter = cv2.arcLength(contour, True)
                approxContour = cv2.approxPolyDP(
                    contour, 0.02*contourPerimeter, True)

                if contourArea > maxContourArea and len(approxContour) == 4:
                    biggestContour = approxContour
                    maxContourArea = contourArea

        return biggestContour

    def reorder(self, contour):
        contour = contour.reshape((4, 2))
        modifiedContour = np.zeros((4, 1, 2), np.int32)

        add = contour.sum(axis=1)
        modifiedContour[0] = contour[np.argmin(add)]
        modifiedContour[3] = contour[np.argmax(add)]

        diff = np.diff(contour, axis=1)
        modifiedContour[1] = contour[np.argmin(diff)]
        modifiedContour[2] = contour[np.argmax(diff)]

        return modifiedContour

    def getTopView(self, image, biggestContour):
        '''
        
        '''
        pts1 = np.float32(biggestContour)
        pts2 = np.float32(
            [[0, 0],
             [GenerateDocument.IMAGE_WIDTH, 0],
             [0, GenerateDocument.IMAGE_HEIGHT],
             [GenerateDocument.IMAGE_WIDTH, GenerateDocument.IMAGE_HEIGHT]]
        )

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        warpedImage = cv2.warpPerspective(image, matrix, (
            GenerateDocument.IMAGE_WIDTH,
            GenerateDocument.IMAGE_HEIGHT
        ))

        outputImage = cv2.resize(warpedImage, (
            GenerateDocument.IMAGE_WIDTH,
            GenerateDocument.IMAGE_HEIGHT
        ))

        return outputImage

    def convertToBlackAndWhite(self, image):
        '''
        This method converts the image to gray scale.
        '''
        try:
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            binI = cv2.threshold(grayImage, 170, 255, cv2.THRESH_BINARY)
            adapI = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 15)
            return np.array(cv2.addWeighted(src1=binI,alpha=0.7,src2=adapI,beta=0.3,gamma=0))
        except:
            raise Exception('Unsuccessful in color formation')

    def generate(self, image, colorProfile :int=0) -> np.ndarray:
        ''''''
        if type(image) != np.ndarray:
            raise TypeError("image must be a numpy array")

        image = cv2.resize(image,
                                (
                                    GenerateDocument.IMAGE_WIDTH,
                                    GenerateDocument.IMAGE_HEIGHT,
                                ))

        processedImage = self.preProcessing(image)
        contour = self.getBiggestContour(processedImage)
        modifiedContour = self.reorder(contour)
        outputImage = self.getTopView(image, modifiedContour)
        
        # color
        if colorProfile == 0:
            return cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)
        
        # black and white
        elif colorProfile == 2:
            return self.convertToBlackAndWhite(outputImage)
        
        # gray scale
        elif colorProfile == 1:
            return cv2.cvtColor(outputImage, cv2.COLOR_BGR2GRAY)
        
        else:
            raise ValueError("colorProfile must be 'color', 'black_and_white' or 'gray'")