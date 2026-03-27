import json
from tensorflow import keras
import numpy as np

(_, __), (X_test, y_test) = keras.datasets.mnist.load_data()
X_test_cnn = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test_onehot = keras.utils.to_categorical(y_test, 10)

cnn = keras.models.load_model('mnist_model.h5')
_, cnn_acc = cnn.evaluate(X_test_cnn, y_test_onehot, verbose=0)
cnn_acc = round(cnn_acc * 100, 2)
print(f"CNN: {cnn_acc}%")

results = [
    {
        "name": "Logistic Regression",
        "accuracy": 92.66,
        "train_time_sec": 168,
        "parameters": 7850,
        "description": "Simple linear classifier with no hidden layers."
    },
    {
        "name": "Dense Neural Network",
        "accuracy": 98.07,
        "train_time_sec": 134,
        "parameters": 269962,
        "description": "Fully connected layers. Ignores spatial structure of images."
    },
    {
        "name": "CNN",
        "accuracy": cnn_acc,
        "train_time_sec": 600,
        "parameters": cnn.count_params(),
        "description": "Learns spatial features like edges and curves. Best for images."
    }
]

with open('model_comparison.json', 'w') as f:
    json.dump(results, f)

print("Saved to model_comparison.json")
print(json.dumps(results, indent=2))