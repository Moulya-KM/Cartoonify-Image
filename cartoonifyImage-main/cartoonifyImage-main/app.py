
import numpy as np
import streamlit as st
import pytesseract
from PIL import Image #python Imaging library, to open image, streamlit does not support cv2 directly

import cv2
#import numpy as np

import matplotlib.pyplot as plt

# def read_file(filename):
#   img=cv2.imread(filename)
#   img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#   plt.imshow(img)
#   #plt.axes('off')
#   plt.show()
#   return img

def edge_mask(img,line_size,blur_value):

  gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
  gray_blur = cv2.medianBlur(gray,blur_value)

  edges=cv2.adaptiveThreshold(gray_blur,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)

  return edges

def color_quantization(img , k ):

    #transform img
    data=np.float32(img).reshape((-1,3))

    #Determine citeria                                             iteration,accuracy
    criteria = (cv2.TERM_CRITERIA_EPS+ cv2.TERM_CRITERIA_MAX_ITER, 20 , 0.001)

    #implementing k-means (clustering)
    ret , label , center = cv2.kmeans(data , k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(img.shape)

    return result
    


def cartoon(blurred,edges):
  c = cv2.bitwise_and(blurred,blurred,mask=edges)
  return c
  # plt.title("Cartoon")
  # plt.imshow(c)
  # plt.show()

  # plt.title("org_img")
  # plt.imshow(org_img)
  # plt.show()

def cartoonization(img):
  line_size=5

  blur_value = st.sidebar.slider('Sharpness of image (the lower the value, the sharper it is)', 1, 99, 25, step=2)
  k = st.sidebar.slider('Tune the color averaging effects (low: only similar colors will be smoothed, high: dissimilar color will be smoothed)', 2, 100,9,step=2)
  d = st.sidebar.slider('Tune the smoothness level of the image (the higher the value, the smoother the image)', 3, 99, 3, step=2)

  edges=edge_mask(img,line_size,blur_value)
  img_quantize=color_quantization(img , k )
  blurred = cv2.bilateralFilter(img_quantize,d,sigmaColor=200,sigmaSpace=200)
  cc=cartoon(blurred,edges)
  return cc

##################

#pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract' #configuration command for heroku stack 
st.set_option('deprecation.showfileUploaderEncoding',False) #to warning ignore
st.title("Cartoonify Images")  #print title and text
st.text("Upload the Image")

uploaded_file=st.sidebar.file_uploader("Choose an image...",type=['jpg','png','jpeg'])
if uploaded_file is not None:
  img=Image.open(uploaded_file)  #reads the image, similar cv2.imread
  image=np.array(img)
  st.image(image,caption="Uploaded Image",use_column_width=True) #displays the image in its actual size 
  st.write("")  #print blank space

  

  #if st.button("Cartoonify"):  #creates a button called as predict
  st.write("Cartoon image")   
  cartoon=cartoonization(image)
  st.image(cartoon,caption="Cartooned Image",use_column_width=True)
    # op=pytesseract.image_to_string(img)  #pytesseract converts img to text and prints
    # st.title(op)
