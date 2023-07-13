import streamlit as st
import os
import cv2
from PIL import Image
from mtcnn import MTCNN
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace



st.title("Which bollywood celebrity are you")

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector = MTCNN()
feature_list = pickle.load(open('embeddings.pkl','rb'))
filenames=pickle.load(open('filename.pkl','rb'))

def save_uploaded_image(uploaded_image):#this will save users img in uploads folder
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())#will write img and all buffer data to file
        return True
    except:
        return False

def extract_features(img_path,model,detector):
    img= cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']  # Extract details of face from facebox attribute
    face = img[y:y + height, x:x + width]  # Cropping face
    image = Image.fromarray(face)  # creates PIL image
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array,axis=0)  # extra dimension will be added and single image becomes a batch of images
    preprocess_img = preprocess_input(expanded_img)
    result = model.predict(preprocess_img).flatten()
    return result

def recommend(features,feature_list):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

uploaded_img = st.file_uploader('Choose an image')
if uploaded_img is not None:
    #save img in a directory
    if save_uploaded_image(uploaded_img):
        #load image
        display_image= Image.open(uploaded_img) #PIL image


        #extract features
        features=extract_features(os.path.join('uploads',uploaded_img.name),model,detector)

        #recommend
        index_pos=recommend(features,feature_list)

        col1, col2 = st.columns(2)

        with col1:
            st.header('Your uploaded image')
            st.image(display_image)
        with col2:
            st.header("Seems like "+ " ".join(filenames[index_pos].split('\\')[1].split('_')))
            st.image(filenames[index_pos], width=300)