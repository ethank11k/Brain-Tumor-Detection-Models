def image_cropping(set_input, pixel_val=0):
    
    set_output = []
    
    for img in set_input:
        
        gray_color = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_color = cv2.GaussianBlur(gray_color, (5, 5), 0)

        threshold = cv2.threshold(gray_color, 45, 255, cv2.THRESH_BINARY)[1]
        threshold = cv2.erode(threshold, None, iterations=2)
        threshold = cv2.dilate(threshold, None, iterations=2)

        contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        c = max(contours, key=cv2.contourArea)

        extreme_left = tuple(c[c[:, :, 0].argmin()][0])
        extreme_right = tuple(c[c[:, :, 0].argmax()][0])
        extreme_top = tuple(c[c[:, :, 1].argmin()][0])
        extreme_bot = tuple(c[c[:, :, 1].argmax()][0])

        pixels_add = pixel_val
        
        new_image = img[extreme_top[1]-pixels_add:extreme_bot[1]+pixels_add, extreme_left[0]-pixels_add:extreme_right[0]+pixels_add].copy()
        
        set_output.append(new_image)

    return np.array(set_output)
    

def image_preprocess(set_input, image_size):
    set_output = []
    for img in set_input:
        img = cv2.resize(
            img,
            dsize=image_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_output)
    
    
TRAIN_DIR = 'TRAIN_CROP/'
VAL_DIR = 'VAL_CROP/'
    
train_data = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

val_data = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_generator = train_data.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)


validation_generator = val_data.flow_from_directory(
    VAL_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)