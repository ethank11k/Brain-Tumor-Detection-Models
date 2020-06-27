concatenate_layer = keras.layers.concatenate([inception_output, vgg_output])
#concatenate_layer = keras.layers.concatenate([inception_output, vgg_output,resnet_output])
concatenate_layer = Dense(256, kernel_initializer=keras.initializers.glorot_uniform(seed=66), activation = 'sigmoid')(concatenate_layer)
concatenate_layer = Dense(1, kernel_initializer=keras.initializers.glorot_uniform(seed=66), activation = 'sigmoid')(concatenate_layer)


final_model = Model(inputs = input_img, outputs = concatenate_layer)


final_model.layers[0].trainable = False
final_model.layers[1].trainable = False
final_model.layers[2].trainable = False
#final_model.layers[3].trainable = False

final_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



EPOCHS = 150
es =                                                                                                                                                                                     EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=10
)

history = final_model.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)