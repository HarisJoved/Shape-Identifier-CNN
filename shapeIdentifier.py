import os
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import json


class ShapeClassifierGUI:
    def __init__(self, root, model_path, class_indices_path):
        self.root = root
        self.root.title("Shape Classifier")
        
        # Load the model and class indices
        try:
            self.model = load_model(model_path)
        except Exception as e:
            raise FileNotFoundError(f"Error loading model: {e}")
        
        try:
            with open(class_indices_path, "r") as f:
                self.class_indices = json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Error loading class indices: {e}")

        self.classes = {v: k for k, v in self.class_indices.items()}  # Reverse mapping

        # Canvas variables
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # Initialize UI components
        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=400, height=400, bg='white', bd=2, relief='solid')
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Buttons
        ttk.Button(button_frame, text="Clear Canvas", command=self.clear_canvas).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Upload Image", command=self.upload_image).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Predict Shape", command=self.predict_shape).grid(row=0, column=2, padx=5)
        
        # Prediction label
        self.prediction_label = ttk.Label(main_frame, text="Draw or upload a shape to predict")
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Create PIL image for drawing
        self.image = Image.new('RGB', (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.drawing:
            if self.last_x and self.last_y:
                self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, 
                                     width=2, fill='black', smooth=True)
                self.draw.line([self.last_x, self.last_y, event.x, event.y], 
                             fill='black', width=2)
            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('RGB', (400, 400), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw or upload a shape to predict")

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            try:
                # Load and resize image to fit canvas
                uploaded_image = Image.open(file_path)
                uploaded_image = uploaded_image.resize((400, 400), Image.Resampling.LANCZOS)
                
                # Clear canvas and display uploaded image
                self.clear_canvas()
                self.image = uploaded_image
                photo = ImageTk.PhotoImage(uploaded_image)
                self.canvas.create_image(0, 0, anchor='nw', image=photo)
                self.canvas.image = photo  # Keep a reference
            except Exception as e:
                self.prediction_label.config(text=f"Error uploading image: {e}")

    def preprocess_image(self):
        """
        Preprocess the image to match the model's input requirements.
        """
        # Resize to model's input size (128x128)
        img = self.image.resize((128, 128))
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)

    def predict_shape(self):
        """
        Predict the shape based on the drawn or uploaded image.
        """
        try:
            # Preprocess the image
            processed_image = self.preprocess_image()
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_class = np.argmax(predictions)
            predicted_label = self.classes[predicted_class]
            
            # Update the prediction label
            self.prediction_label.config(text=f"Predicted Shape: {predicted_label}")
        except Exception as e:
            self.prediction_label.config(text=f"Error during prediction: {e}")


def main():
    # Ensure current directory is set correctly
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Paths to model and class indices
    model_path = "saved_model.h5"
    class_indices_path = "class_indices.json"

    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"Class indices file not found: {class_indices_path}")

    # Launch the GUI
    root = tk.Tk()
    app = ShapeClassifierGUI(root, model_path, class_indices_path)
    root.mainloop()


if __name__ == "__main__":
    main()
