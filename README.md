# Skin Type Classification and Acne Detection

This project is focused on developing machine learning models to classify skin types and detect acne. It leverages deep learning models and computer vision techniques to analyze facial images for dermatological insights.

## Project Structure

The following describes the directory structure and key files of the project:

### Directories
- **`acne_detection/`**: Contains resources, scripts, and models specific to acne detection.
- **`faceshape/`**: Includes tools and data for facial shape analysis.
- **`config/`**: Stores configuration files for the project.
- **`output/`**: Stores the outputs such as model predictions and logs.
- **`MCDVT/`**: Related to additional datasets or modules.

### Key Files

#### Models
- **`best_model.h5`**: Trained model for skin type classification.
- **`best_resnet50_skin_type.pth`**: PyTorch implementation of a ResNet50 model for skin classification.
- **`resnet_skin_classification.h5`**: ResNet-based skin classification model.
- **`skin_type_classification_model.h5`**: A secondary trained model for skin type classification.
- **`transfer_learning_model.h5`**: Model trained using transfer learning techniques.
- **`vgg16_skin_type_classifier.h5`**: VGG16-based model for skin type classification.

#### Scripts
- **`augment.py`**: Script for data augmentation.
- **`backend_test.py`**: Backend testing script.
- **`createdf.py`**: Script to create datasets from raw images.
- **`preprocess.py`**: Preprocessing script for images.
- **`torch_train.py`**: Training script using PyTorch.
- **`train.py`**: General training script for TensorFlow/Keras models.
- **`transfer_learning.py`**: Script for implementing transfer learning.
- **`vgg_train.py`**: Training script for the VGG16 model.
- **`torch_test.py`**: PyTorch model testing script.
- **`test_image.py`**: Script to test model predictions on a single image.

#### Other Files
- **`haarcascade_frontalface_default.xml`**: Haar Cascade XML file for face detection.
- **`README.dataset`**: Notes and details about the dataset.
- **`README.roboflow`**: Instructions and setup for Roboflow integration.

### Images
- **`KKK.jpg, Oil.jpg, Oil2.jpg, THS.jpg`**: Sample images used for testing and validation.
- **`test.jpg`**: Test image for model validation.

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.10.0+
- PyTorch 1.10.0+
- OpenCV
- Required Python libraries (install using `requirements.txt`):
  ```
  pip install -r requirements.txt
  ```

## Usage

### Skin Type Classification
1. Prepare the dataset and place it in the `data/` directory.
2. Train the model:
   ```
   python train.py
   ```
3. Test the model on a sample image:
   ```
   python test_image.py --image_path test.jpg
   ```

### Acne Detection
1. Use the scripts in the `acne_detection/` folder to process images.
2. Train and test models:
   ```
   python acne_detection/train.py
   python acne_detection/test.py
   ```

### Transfer Learning
1. Fine-tune the `transfer_learning_model.h5` for custom datasets:
   ```
   python transfer_learning.py
   ```

## Dataset
- Datasets for skin type classification and acne detection should be organized in the `data/` folder.
- Use `README.dataset` and `README.roboflow` for detailed dataset setup instructions.

## Models
- Use pre-trained models stored in `.h5` or `.pth` files for testing and fine-tuning.
- Modify configuration in `config/` for custom experiments.

## Output
- Model outputs and logs are stored in the `output/` directory.
- Evaluate performance metrics such as accuracy, precision, and recall.

## Future Improvements
- Integrate real-time skin analysis using webcam input.
- Expand functionality for detecting other dermatological conditions.

## License
This project is licensed under the MIT License. See `LICENSE` for more information.

## Acknowledgments
- Open-source contributions from TensorFlow, PyTorch, and OpenCV.
- Roboflow for dataset preparation tools.

