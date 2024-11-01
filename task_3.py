import os
import cv2
import numpy as np
import random
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Define data paths
CAT_DIR = '/kaggle/input/microsoft-catsvsdogs-dataset/PetImages/Cat'
DOG_DIR = '/kaggle/input/microsoft-catsvsdogs-dataset/PetImages/Dog'

# Define whether to include test split or not
INCLUDE_TEST = True
split_ratio = 0.9  # 90% train-validation, 10% test

# Set image size and initialize data
img_size = 150
data = []
class_names = ['Cat', 'Dog']

# Load and resize an image
def load_and_resize_image(image_path, size=(128, 128)):
    try:
        # Load the image
        img = cv2.imread(image_path)
        
        # Check if the image is loaded successfully
        if img is None:
            print(f"Warning: Could not load image {image_path}. Skipping.")
            return None
        
        # Resize the image
        img_resized = cv2.resize(img, size)
        return img_resized
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Load and preprocess images with resizing
def load_data(directory, label):
    images = []
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        img = load_and_resize_image(img_path, size=(img_size, img_size))
        if img is not None:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            images.append([img_gray.flatten(), label])       # Flatten and add label
    return images

# Load data for both classes
data.extend(load_data(CAT_DIR, 0))  # 0 for Cat
data.extend(load_data(DOG_DIR, 1))  # 1 for Dog

# Shuffle and split data
random.shuffle(data)
X = np.array([i[0] for i in data]) / 255.0  # Normalize pixel values
Y = np.array([i[1] for i in data])

# Train-validation-test split
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=(1 - split_ratio), random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Train SVM model
svm_model = svm.SVC(kernel='linear', probability=True)  # Using linear kernel
svm_model.fit(X_train, Y_train)

# Evaluate on validation and test sets
def evaluate_model(model, X, Y, dataset_type="Validation"):
    Y_pred = model.predict(X)
    accuracy = accuracy_score(Y, Y_pred)
    print(f"{dataset_type} Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(Y, Y_pred, target_names=class_names))

evaluate_model(svm_model, X_val, Y_val, dataset_type="Validation")
if INCLUDE_TEST:
    evaluate_model(svm_model, X_test, Y_test, dataset_type="Test")

# Plot random images with predictions
def plot_data(X, Y, model, n_images=10):
    indices = random.sample(range(len(X)), n_images)
    plt.figure(figsize=(14, 10))
    for i, idx in enumerate(indices):
        image = X[idx].reshape(img_size, img_size)
        true_label = class_names[Y[idx]]
        pred_label = class_names[model.predict([X[idx]])[0]]
        plt.subplot(3, 4, i+1)
        plt.imshow(image, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    plt.show()

plot_data(X_val, Y_val, svm_model, n_images=7)
if INCLUDE_TEST:
    plot_data(X_test, Y_test, svm_model, n_images=7)
