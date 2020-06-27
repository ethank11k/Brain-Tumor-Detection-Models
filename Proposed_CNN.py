input_img = Input(shape=(224, 224, 3), name='main_input')

cnn_output = ZeroPadding2D((2, 2))(input_img)     
cnn_output = Conv2D(32, (7, 7), strides = (1, 1))(cnn_output)
cnn_output = BatchNormalization(acnn_outputis = 3)(cnn_output)
cnn_output = Activation('relu')(cnn_output)   
cnn_output = Macnn_outputPooling2D((4, 4))(cnn_output) 


cnn_output = Conv2D(32, (7, 7), strides = (1, 1))(cnn_output)
cnn_output = BatchNormalization(acnn_outputis = 3)(cnn_output)
cnn_output = Activation('relu')(cnn_output)   
cnn_output = Macnn_outputPooling2D((4, 4))(cnn_output)

cnn_output = Flatten()(cnn_output) 
cnn_output = Dense(1, activation='sigmoid')(cnn_output) 

final_cnn_model = Model(inputs=input_img, outputs=cnn_output)
final_cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


EPOCHS = 150

es =                                                                                                                                                                                     EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=10
)

history = final_cnn_model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks = [es]
)

