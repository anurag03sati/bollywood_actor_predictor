from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list=np.array(pickle.load(open('embeddings.pkl','rb')))


filenames=pickle.load(open('filename.pkl','rb'))

model = VGGFace(model='resnet50',include_top=False,input_shape=(224,224,3),pooling='avg')
detector = MTCNN()

#loading image
sample_img=cv2.imread('sample/srk2.jpeg')
results = detector.detect_faces(sample_img)

if len(results) > 0:
    x, y, width, height = results[0]['box']  # Extract details of face from facebox attribute
    face = sample_img[y:y+height, x:x+width]  # Cropping face
    #cv2.imshow('output', face)
    #cv2.waitKey(0)

    # Close OpenCV windows and release resources
    #cv2.destroyAllWindows()
else:
    print("No faces were detected in the image.")

#extract features of img

image=Image.fromarray(face)#creates PIL image
image=image.resize((224,224))
face_array = np.asarray(image)
face_array=face_array.astype('float32')
expanded_img=np.expand_dims(face_array,axis=0) #extra dimension will be added and single image becomes a batch of images
preprocess_img=preprocess_input(expanded_img)
result = model.predict(preprocess_img).flatten() #so that results are in 1D
#print(result)
#print(result.shape)


#now compare this one predicted vector with the rest of 8000 vectors and find the perfect match img

#print(cosine_similarity(result.reshape(1,-1),feature_list[0].reshape(1,-1))[0][0])
similarity=[]
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])#this will find out similarity of our predicted img with rest 8000 img and store in similarity
#print(len(similarity))
#list(enumerate(similarity)) basically creates a list of tuples with distance corresponding to which image index

#print(sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])) sorts in descending order and gives best match tuples in order since 1 means cos 0 thus closest
index_pos=sorted(list(enumerate(similarity)),reverse=True,key=lambda x:x[1])[0][0]

#display img

temp_img=cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)