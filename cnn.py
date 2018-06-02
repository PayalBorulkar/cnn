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
                         steps_per_epoch=1000,
                         epochs=5,
                         validation_data=test_set,
                         validation_steps=1000)


classifier.save('Image_Classifying.h5')
classifier.save_weights('my_model_weight.h5')

