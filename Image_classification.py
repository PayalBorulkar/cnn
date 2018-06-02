from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import sys

#Loading saved model
new_model=load_model('Image_Classifying.h5')
new_model.get_weights()
#new_model.summary()

# Please enter image path e.g c:\Desktop\mountain.jpg. In dataset folder in single prediction folder some images are present
print("Enter image path")
x=input()

#Evaluation Tool for classifying image
test_image=image.load_img(x,target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=new_model.predict(test_image)
if result[0][0]==1:
    prediction='Outdoor'
else:
    prediction='Indoor'
print("Predicted class label: ",prediction)
