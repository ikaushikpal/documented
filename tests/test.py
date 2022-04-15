from pathlib import Path
import cv2
import numpy as np

try:
    from src.generateDocument import GenerateDocument
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "genDoc", Path(__file__).parent.parent/"src"/ "generateDocument.py",submodule_search_locations = [])
    genDoc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(genDoc)

try:
    from src.pdf import PDF
except ModuleNotFoundError:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pdf", Path(__file__).parent.parent/"src"/ "pdf.py",submodule_search_locations = [])
    pdf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdf)
# ##################################
# image = cv2.imread('documented/tests/upload/scanned.png')
# # print(type(image))
# gd = genDoc.GenerateDocument()

# page1 = gd.generate(image, 0)
# page2 = gd.generate(image,1)

# p = pdf.PDF([page1,page2])
# buffer = p.convert()
# ###################################

####################################
img2 = cv2.imread('documented/tests/upload/paper2.jpg')
img3 = cv2.imread('documented/tests/upload/paper3.jpg')
img4 = cv2.imread('documented/tests/upload/paper4.jpg')
img1 = cv2.imread('documented/tests/upload/paper1.jpg')

img1 = cv2.resize(img1, (900,750))
img2 = cv2.resize(img2, (900,750))
img3 = cv2.resize(img3, (900,750))
img4 = cv2.resize(img4, (900,750))

gd = genDoc.GenerateDocument()

bytes1 = gd.generate(img1)
bytes2 = gd.generate(img2)
bytes3 = gd.generate(img3)
bytes4 = gd.generate(img4)

p = pdf.PDF([bytes2,bytes3, bytes4])
buffer = p.convert()
####################################
with open('doc2.pdf','wb') as pdf:
    pdf.write(buffer)
# cv2.imshow('Page', page )
cv2.waitKey()
cv2.destroyAllWindows()