# import os
# import pickle
# actors=os.listdir('data')
# #print(actors)
# filename=[]
# for actor in actors:
#     for file in os.listdir(os.path.join('data',actor)):
#         filename.append(os.path.join('data',actor,file))
# print(filename)
# print(len(filename))
# pickle.dump(filename,open('filename.pkl','wb'))

import tensorflow as tf

from keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('filename.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
#print(model.summary())

def feature_extractor(img_path,model):
    img=image.load_img(img_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)#extra dimension will be added and single image becomes a batch of images
    preprocessed_img=preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()#all features stored here
    return result

features=[]
for file in tqdm(filenames):
    features.append(feature_extractor(file,model))# this will extract 2048  features of every image in a vector and we'll append all these vectors(8000) in features

pickle.dump(features,open('embeddings.pkl','wb'))

