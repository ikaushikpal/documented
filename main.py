import streamlit as st
from src.pdf import PDF
from src.generateDocument import GenerateDocument
import cv2
import numpy as np


class Image:
    def __init__(self, id, name, size, data):
        self.id = id
        self.name = name
        self.size = size
        self.data = data


def preview(images):
    N = 5
    cols = st.columns(N)
    for i, image in enumerate(images):
        cols[i % N].image(image.data, use_column_width=True,
                          caption=image.name)

@st.cache
def loadObjects():
    return GenerateDocument()

colorMapper = {
    'Color': 0,
    'Black and White': 2,
    'Gray Scale': 1
}
images = []
generateDocObject = loadObjects()

st.title("Document Generator")

with st.sidebar:
    uploaded_files = st.file_uploader(
        "Choose images",
        type=['jpg', 'png', 'jpeg', 'gif', 'bmp'],
        accept_multiple_files=True,
        help='help needed',
        key='file_uploader', )

    images.clear()
    for uploaded_file in uploaded_files:
        image = Image(uploaded_file.id,
                      uploaded_file.name,
                      uploaded_file.size,
                      uploaded_file.read())
        images.append(image)


if len(images) == 0:
    st.write('No images to show preview')
else:
    preview(images)
    fileName = st.text_input('Enter file name',
                             max_chars=100,
                             help='help needed',
                             placeholder='fileName'
                             )
    fileName = fileName.strip()
    if fileName == '':
        st.error('File name cannot be empty')
    
    if fileName[-4:] != '.pdf':
        fileName += '.pdf'
    
    colorProfile = colorMapper[st.radio(
        "color profile of pdf",
        ('Color', 'Gray Scale', 'Black and White'),
        index=0)]

    if st.button('Generate'):
        convertedImages = []
        percent_complete = 0
        my_bar = st.progress(percent_complete)

        for i in range(len(images)):
            if i == len(images)-1:
                percent_complete = 100
            else:
                percent_complete += round(100/len(images))

            ndImage = np.frombuffer(images[i].data, dtype=np.uint8)
            buffImage = cv2.imdecode(ndImage, cv2.IMREAD_COLOR)
            bgrImages = cv2.cvtColor(buffImage,
                                     cv2.COLOR_RGB2BGR)
            try:
                convertedImages.append(generateDocObject.generate(bgrImages))
            except Exception as e:
                st.write(f'Can not convert image {images[i].name}')

            my_bar.progress(percent_complete)

        pdfObj = PDF(convertedImages)
        finalPdf = pdfObj.convert()
        st.success('Generated')

        st.download_button('Download PDF',
                           data=finalPdf,
                           file_name=fileName,
                           mime='bytes',
                           help='help needed')
