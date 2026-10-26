# 🌾 Agriculture Land Classifier

A convolutional neural network (CNN) built with TensorFlow/Keras that classifies images as **[CLASS A]** or **[CLASS B]** <!-- e.g. agricultural vs non-agricultural land -->.

## Results

| Metric | Score |
|---|---|
| Validation accuracy | **XX%** |

<!-- Add accuracy/loss curves and a few sample predictions here -->
<!-- ![Training curves](images/training_curves.png) -->

## Dataset

<!-- Describe the dataset: where it's from (link), how many images, image source (satellite / aerial / ground photos), and the class split -->

- **Source:** [dataset name and link]
- **Images:** [number] images, resized to 128 × 128 RGB
- **Classes:** [CLASS A], [CLASS B]

## Model architecture

```
Input (128 × 128 × 3)
→ Conv2D(32, ReLU)  → MaxPooling
→ Conv2D(64, ReLU)  → MaxPooling
→ Conv2D(128, ReLU) → MaxPooling
→ Flatten
→ Dense(128, ReLU) → Dropout
→ Dense(1, Sigmoid)
```

Three convolution blocks extract features of increasing complexity (edges → textures → field and land patterns). Dropout reduces overfitting, and a single sigmoid output gives the probability of the positive class.

## Files

| File | Description |
|---|---|
| `train.ipynb` | Data loading, augmentation, training and evaluation <!-- add your training notebook --> |
| `best_agriculture_model.keras` | Checkpoint with the best validation score |
| `final_agriculture_classifier.keras` | Model after the final epoch |

## Use the model

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image

model = keras.models.load_model("best_agriculture_model.keras")

img = image.load_img("test.jpg", target_size=(128, 128))
x = image.img_to_array(img) / 255.0   # use the same scaling as in training
x = np.expand_dims(x, axis=0)

prob = model.predict(x)[0][0]
print("CLASS B" if prob >= 0.5 else "CLASS A", f"({prob:.2f})")
```

## Tech stack

Python · TensorFlow / Keras · NumPy · Matplotlib

## Possible improvements

- Transfer learning (MobileNetV2 or EfficientNet) for higher accuracy with fewer images
- Data augmentation (flips, rotations, brightness)
- Grad-CAM heatmaps to show which parts of the image drive the prediction
- A Gradio demo on Hugging Face Spaces

---

Built by [Yashneil Purkayastha](https://www.linkedin.com/in/yashneil-purkayastha-87a593371)
