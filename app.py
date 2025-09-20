import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Define the folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model and class names
try:
    model = load_model('plant_disease_model.keras')
    with open('class_names.pkl', 'rb') as f:
        class_names = pickle.load(f)
    print("Model and class names loaded successfully!")
except Exception as e:
    print(f"Error loading model or class names: {e}")
    model = None
    class_names = []

def preprocess_image(image):
    """Preprocesses a raw image to match the model's input requirements."""
    # Resize the image
    image = image.resize((128, 128))
    # Convert to numpy array
    img_array = np.array(image)
    # Normalize to [0, 255] (assuming model was trained on this range)
    img_array = img_array.astype('float32')
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# The main page route
@app.route('/')
def home():
    return render_template('index.html')

# The prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file to a temporary location
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Open and preprocess the image
            image = Image.open(filepath).convert('RGB')
            processed_image = preprocess_image(image)
            
            # Make a prediction
            prediction = model.predict(processed_image)
            
            # Get the predicted class and confidence
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index])
            
            # Return the result
            return jsonify({
                "predicted_class": predicted_class_name,
                "confidence": confidence
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the saved file
            if os.path.exists(filepath):
                os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)