import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Conv2D

input_img = Input(shape=(224, 224, 3), name='main_input')



vgg_model = VGG16(include_top=False,weights="imagenet", input_shape=(224, 224, 3))(input_img)
vgg_model = Flatten()(vgg_model)
vgg_model = Dropout(0.5)(vgg_model)
vgg_output = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=66), activation='sigmoid')(vgg_model)

vgg_model_final = Model(inputs=input_img, outputs=vgg_output)



resnet_model = ResNet50(include_top=False, weights = "imagenet", input_shape=(224, 224, 3))(input_img)
resnet_model = Flatten()(resnet_model)
resnet_model = Dropout(0.5)(resnet_model)
resnet_output = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=66),  activation = 'sigmoid')(resnet_model)

resnet_model_final = Model(inputs=input_img, outputs=resnet_output)



inception_model = InceptionV3(include_top=False, weights = "imagenet", input_shape=(224, 224, 3))(input_img)
inception_model = Flatten()(inception_model)
inception_model = Dropout(0.5)(inception_model)
inception_output = Dense(256,kernel_initializer=keras.initializers.glorot_uniform(seed=66),  activation = 'sigmoid')(inception_model)

inception_model_final = Model(inputs=input_img, outputs=inception_output)

