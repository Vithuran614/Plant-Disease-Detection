# Plant Disease Detection 🌱

**Project Overview:**
This project is a deep learning model that identifies diseases in plants based on images of their leaves. The goal is to provide a fast and accurate diagnosis to help gardeners and farmers take timely action. The solution includes a web application built with Flask, allowing users to upload a leaf image and receive an immediate diagnosis.

## Key Features:

**Disease Classification**: Identifies and classifies 15 different plant diseases, as well as healthy leaves.

**Web Application**: A user-friendly interface built with Flask that handles image uploads and displays predictions.

**Deep Learning Model**: Utilizes a powerful convolutional neural network for high-accuracy predictions.

## Model Architecture:

The model is built using a transfer learning approach with a pre-trained ResNet50 convolutional neural network. The base model's weights were pre-trained on the ImageNet dataset and its layers are frozen, acting as a powerful feature extractor. New layers were added to adapt the model for plant disease classification.

* **Base Model**: **ResNet50**, pre-trained on the `imagenet` dataset with an input shape of `(128, 128, 3)`. The base model's layers are set to `trainable=False`.
* **Custom Layers**:
  * `Flatten` layer to convert the base model's output into a 1D feature vector.
  * `Dense` layer with **512 neurons** and a `relu` activation function.
  * `Dropout` layer with a rate of **0.2** for regularization.
  * `Dense` output layer with **15 neurons** and a `softmax` activation function for multi-class classification.
* **Compilation**:
  * **Optimizer**: The model is compiled using the `Adam` optimizer.
  * **Loss Function**: `categorical_crossentropy`.
  * **Metrics**: The model is evaluated based on its `accuracy`.

**Dataset:**

The model was trained on the PlantVillage Dataset, a publicly available collection of 20,000+ images of plant leaves. The dataset contains images of both healthy and diseased leaves across fifteen distinct classes.

## Getting Started

### Installation:

1.Clone this repository to your local machine:
``git clone https://github.com/[YourUsername]/Plant-Disease-Detection.git``

2.Navigate to the project directory:
``cd Plant-Disease-Detection``

3.Install the required Python libraries:
``pip install -r requirements.txt``

### How to Run the Application

1.Start the Flask server:
``python app.py``

2.Open your web browser and navigate to http://127.0.0.1:5000 to access the application.



## Results:

The model achieved the following performance metrics during training and evaluation:

Training Accuracy: 96.59%

Validation Accuracy: 94.33%

Test Accuracy: 95.00%

**Technologies Used:**

* Python
* TensorFlow / Keras
* Flask
* NumPy
* Pillow
* Git & GitHub
* Git Large File Storage (Git LFS) for handling the model file.

**Contact:**

GitHub: https://github.com/Vithuran614

LinkedIn: https://www.linkedin.com/in/vithuran-kailash-673924367/

