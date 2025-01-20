import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2  # Import OpenCV for face detection

# Parameters
EPOCHS = 20
LR = 0.1
STEP = 15
GAMMA = 0.1
BATCH = 16
OUT_CLASSES = 3
IMG_SIZE = 224
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"  # Download the Haar Cascade for face detection

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + FACE_CASCADE_PATH)

# Face detection function
def detect_and_crop_face(image):
    # Convert image back to 8-bit format (0-255)
    image_uint8 = (image * 255).astype(np.uint8)
    
    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If a face is detected, crop to the face region
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Taking the first detected face
        face = image_uint8[y:y+h, x:x+w]
        # Resize to the model's input size
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        return face
    else:
        # If no face is detected, resize the original image
        return cv2.resize(image_uint8, (IMG_SIZE, IMG_SIZE))


# Modify the ImageDataGenerator to include face detection
class FaceDetectionDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, *args, **kwargs):
        batches = super().flow_from_directory(*args, **kwargs)
        while True:
            batch_x, batch_y = next(batches)
            batch_x_faces = np.zeros((batch_x.shape[0], IMG_SIZE, IMG_SIZE, 3))
            for i in range(batch_x.shape[0]):
                batch_x_faces[i] = detect_and_crop_face(batch_x[i])
            yield batch_x_faces, batch_y

# Load data with face detection
def create_dataset(directory, batch_size, img_size, shuffle=True):
    datagen = FaceDetectionDataGenerator(rescale=1./255, validation_split=0.2)
    data = datagen.flow_from_directory(directory,
                                        target_size=(img_size, img_size),
                                        batch_size=batch_size,
                                        class_mode='sparse',
                                        subset='training' if shuffle else 'validation')
    return data

train_ds = create_dataset("config/train", BATCH, IMG_SIZE)
val_ds = create_dataset("config/valid", BATCH, IMG_SIZE)

# Load and modify EfficientNet
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = Dense(OUT_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

# Compile model
optimizer = SGD(learning_rate=LR)
model.compile(optimizer=optimizer,
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch % STEP == 0 and epoch != 0:
        return lr * GAMMA
    return lr

callback = LearningRateScheduler(scheduler)
steps_per_epoch = 1656 // BATCH
validation_steps = 262// BATCH

# Training loop with steps_per_epoch
best_val_acc = 0
for epoch in range(EPOCHS):
    history = model.fit(train_ds,
                        epochs=1,
                        validation_data=val_ds,
                        steps_per_epoch=steps_per_epoch,  # Add this
                        validation_steps=validation_steps,  # Add this
                        callbacks=[callback])
    
    # Extract metrics from the history
    train_loss = history.history['loss'][0]
    train_acc = history.history['accuracy'][0]
    val_loss = history.history['val_loss'][0]
    val_acc = history.history['val_accuracy'][0]
    
    # Check for best accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = model.get_weights()
    
    print(f"Epoch {epoch+1}/{EPOCHS} train loss {train_loss:.4f} acc {train_acc:.4f} val loss {val_loss:.4f} acc {val_acc:.4f}")

# Load best model
model.set_weights(best_model_weights)
# Training loop
