import argparse
import tensorflow as tf
import numpy as np
import json
from PIL import Image

def process_image(image_path):
    """Load and preprocess the image for model prediction."""
    image = Image.open(image_path)
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0   
    return np.expand_dims(image, axis=0) 

def predict(image_path, model, top_k=5):
    """Make predictions using the trained model."""
    image = process_image(image_path)
    preds = model.predict(image)[0]
    top_indices = np.argsort(preds)[-top_k:][::-1]  
    top_probs = preds[top_indices]
    return top_probs, top_indices

def load_class_names(json_path):
    """Load class labels from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Flower Image Classifier")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to trained model")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top predictions")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to names")

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})

    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        class_names = load_class_names(args.category_names)
        class_labels = [class_names.get(str(cls), "Unknown") for cls in classes]
    else:
        class_labels = classes

    print("Top Predictions:")
    for i in range(len(probs)):
        print(f"{class_labels[i]}: {probs[i]:.4f}")

if __name__ == "__main__":
    main()
