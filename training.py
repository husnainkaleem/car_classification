import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

train_path = "C:\\Users\\AK\\Documents\\car_color_dataset\\train"
test_path = "C:\\Users\\AK\Documents\\car_color_dataset\\test"
val_path = "C:\\Users\\AK\\Documents\\car_color_dataset\\val"


# The code essentially defines two functions: preprocess_images, which reads, resizes, and
# normalizes images, and preprocess_dataset, which processes images and assigns labels
# based on class indices. This set of functions can be used to preprocess the images and
#
def preprocess_images(image_paths, target_size=(224, 224)):
    processed_images = []
    for path in image_paths:
        image = cv2.imread(path)
        resized_image = cv2.resize(image, target_size)
        normalized_image = resized_image / 255.0
        processed_images.append(normalized_image)
    return processed_images


def preprocess_dataset(dataset_path):
    classes = os.listdir(dataset_path)
    images = []
    labels = []
    for i, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        class_images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        images.extend(class_images)
        labels.extend([i] * len(class_images))  # Assign label based on class index
    return preprocess_images(images), labels


# Preprocess train, test, and val datasets
X_train, y_train = preprocess_dataset(train_path)
print("completed ")
X_test, y_test = preprocess_dataset(test_path)
print("completed ")
X_val, y_val = preprocess_dataset(val_path)
print("completed ")
# Organize the preprocessed data into NumPy arrays
X_train = np.array(X_train)
print("completed")
y_train = np.array(y_train)
print("completed")
X_test = np.array(X_test)
print("completed")
y_test = np.array(y_test)
print("completed")
X_val = np.array(X_val)
print("completed")
y_val = np.array(y_val)
print("completed")
import tensorflow as tf
from tensorflow.keras import layers, models

num_classes = 8
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))  # num_classes is the number of car colors
batch_size = 16
epochs = 10

# Define an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Compile the model with the defined optimizer
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train the model for 1 epoch using the full training dataset
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val))

    # Evaluate the model on the test data
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the model after each epoch
    model.save(f'car_model_epoch{epoch + 1:02d}.h5')

    print("Model saved\n")