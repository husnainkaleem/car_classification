import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
loaded_model = tf.keras.models.load_model('car_model_epoch02.h5')

# Preprocess a new image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)  # Add batch dimension

# Path to the new image you want to predict
new_image_path = 'C:\\Users\\AK\\Documents\\testcar\\black.jpg'

# Preprocess the new image
preprocessed_image = preprocess_image(new_image_path)

# Make a prediction
predictions = loaded_model.predict(preprocessed_image)

# Interpret the prediction
predicted_class = np.argmax(predictions)

# Define a dictionary to map predicted classes to car colors
car_color_map = {
    0: "Black",
    1: "Blue",
    2: "Brown",
    3: "Green",
    4: "Grey",
    5: "Red",
    6: "White",
    7: "Yellow"
}

# Display the predicted class and the probability distribution
print(f"Predicted Class: {predicted_class}")
predicted_color = car_color_map.get(predicted_class, "Unknown")
print(f"Car Color is {predicted_color}")
print("Probability Distribution:", predictions)