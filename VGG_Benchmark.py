NUM_CLASSES = 1

vgg16_benchmark = Sequential()
vgg16_benchmark.add(vgg)
vgg16_benchmark.add(layers.Dropout(0.3))
vgg16_benchmark.add(layers.Flatten())
vgg16_benchmark.add(layers.Dropout(0.5))


vgg16_benchmark.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

vgg16_benchmark.layers[0].trainable = False

vgg16_benchmark.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)
vgg16_benchmark.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0003), metrics=["accuracy"])

vgg16_benchmark.summary()

EPOCHS = 150
es = EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=10
)

history = vgg16_benchmark.fit_generator(
    train_generator,
    steps_per_epoch=50,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25,
    callbacks=[es]
)

