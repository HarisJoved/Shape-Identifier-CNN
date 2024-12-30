# Shape Identifier Using CNN

This project is a Convolutional Neural Network (CNN) based system designed to identify geometric shapes. It uses a trained deep learning model to classify shapes like circles, squares, rectangles, and more, from drawn or uploaded images.

---

## Features

- **Shape Classification:** Recognizes and classifies various geometric shapes.
- **Pre-Trained Model:** Utilizes a trained `saved_model.h5` for efficient predictions.
- **Visualization:** Displays the input and classified shapes for user feedback.
- **User Interaction:** Supports image upload and live drawing for classification.

---

## Project Structure

```plaintext
Shape-Identifier-CNN/
├── README.md
├── saved_model.h5                  # Pre-trained model
├── shape-identifier-ann.ipynb      # Jupyter Notebook for experimentation
├── shapeIdentifier.py              # Main Python script for shape classification
├── class_indices.json              # JSON file for mapping class indices to labels
├── Images                          # Sample input images
│   ├── circle img.png
│   ├── rectangle drawn.png
│   └── ...
```

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Shape-Identifier-CNN.git
   cd Shape-Identifier-CNN
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.7+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   Execute the main script to start the shape classification:
   ```bash
   python shapeIdentifier.py
   ```

---

## Usage

1. **Load an Image:**
   - Upload a pre-saved image of a geometric shape.
   - Draw a shape directly within the app (if supported).

2. **Prediction:**
   - The system will process the image and output the shape classification.

3. **Examples:**
   - Input: `circle drawn.png`  
     Output: `Circle`
     ![Circle Output](circle%20drawn.png)
   - Input: `rectangle img.png`  
     Output: `Rectangle`
     ![Rectangle Output](rectangle%20img.png)
   - Input: `square drawn.png`  
     Output: `Square`
     ![Square Output](square%20drawn.png)
   - Input: `trapezoid drawn.png`  
     Output: `Trapezoid`
     ![Trapezoid Output](trapezoid%20drawn.png)
   - Input: `parallelogram img.png`  
     Output: `Parallelogram`
     ![Parallelogram Output](parallelogram%20img.png)
   - Input: `kite img.png`  
     Output: `Kite`
     ![Kite Output](kite%20img.png)

---

## Dataset

The model was trained on a custom dataset of hand-drawn and computer-generated geometric shapes, categorized into classes such as:

- Circle
- Square
- Rectangle
- Triangle
- Parallelogram
- Rhombus
- Trapezoid

---

## Model

- **Framework:** TensorFlow/Keras
- **File:** `saved_model.h5`
- **Structure:** A simple CNN architecture tailored for image classification.

---


## Contact

For questions or support, please contact:

- **Name:** Haris Javed
- **Email:** [harisjaved16661@gmail.com]
