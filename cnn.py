from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing CNN

classifier=Sequential()

#step-1 convolution layer
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3),activation='relu'))
#step-2 pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Adding 2 convolutional layer
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step-3 Flatten
classifier.add(Flatten())
#step-4 Full Convolution
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim= 1,activation='sigmoid'))

#compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting Cnn to in image(for data preparation)
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                  target_size=(64, 64),
                                                  batch_size=32,
                                                  class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=6000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)

#Making new prediction(Evaluation tool for classifying single image)

import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/indoor_or_outdoor1.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='outdoor'
else:
    prediction='indoor'
print(prediction)
