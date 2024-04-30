import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def prepare_data(classes):
    train_dir = "train_data"
    test_dir = "test_data"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join("Celebrity Faces Dataset", class_name)
        all_images = os.listdir(class_dir)
        selected_images = np.random.choice(all_images, size=min(100, len(all_images)), replace=False)
        train_images, test_images = train_test_split(selected_images, test_size=0.2, random_state=42)
        
        for image in train_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(train_dir, class_name, image)
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            shutil.copy(src, dst)
        
        for image in test_images:
            src = os.path.join(class_dir, image)
            dst = os.path.join(test_dir, class_name, image)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)
            shutil.copy(src, dst)

def create_generators(train_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(150, 150),
            batch_size=20,
            class_mode='categorical')

    return train_generator, test_generator

def visualize_images(generator, num_images=5):
    class_labels = {v: k for k, v in generator.class_indices.items()}
    plt.figure(figsize=(10, 5))
    images, labels = next(generator)  # Get a batch of images and labels
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        image = images[i]
        label = class_labels[np.argmax(labels[i])]  # Use np.argmax to get the label index
        plt.imshow(image)
        plt.title(label)
        plt.axis('off')
    plt.show()