from io import BytesIO

import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow Lite model.
def Bili_Level(image_data,modelType='head'):
      model_path=f'tf_lite_{modelType}_model.tflite'

      interpreter = tf.lite.Interpreter(model_path=model_path)
      interpreter.allocate_tensors()

      # Load and preprocess the image.
      image = Image.open(BytesIO(image_data))
      image = image.resize((480, 480))  # Resize to match model input size.
      image = np.array(image) / 255.0 # Normalize pixel values.
      image = np.expand_dims(image, axis=0) # Add batch dimension.
      image = image.astype(np.float32) # Convert to float32.

      # Run the model on the input image.
      input_details = interpreter.get_input_details()

      output_details = interpreter.get_output_details()
      interpreter.set_tensor(input_details[0]['index'], image)
      interpreter.invoke()
      output_data = interpreter.get_tensor(output_details[0]['index'])

      # Process the output data for regression prediction.
      predicted_value = output_data[0][0] # Get the predicted value from the output tensor.

      # Print the predicted value.
      print("BILIRUBIIN LEVEL IS:", predicted_value, 'mg/dL')

      return predicted_value.item();