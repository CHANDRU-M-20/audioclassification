# Audio Classification for Fan Sounds

## Overview
This project focuses on classifying fan sounds using deep learning techniques. The system extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio files and trains a neural network to classify different fan sound patterns.

## Objectives
- Extract MFCC features from audio recordings.
- Train a deep learning model to classify fan sounds.
- Improve model accuracy with appropriate preprocessing and network architecture.
- Save the trained model for future inference.

## Technologies Used
- **Python**
- **Librosa** (for audio feature extraction)
- **Pandas & NumPy** (for data handling)
- **TensorFlow & Keras** (for deep learning model development)
- **Scikit-learn** (for preprocessing and model evaluation)

## Dataset
- **Audio Data Path:** `Audios/audio/`
- **Metadata File:** `Audios/metadata/FanSounds.csv`
- Each entry in the metadata file contains:
  - `slice_file_name`: Name of the audio file.
  - `class`: Label corresponding to the fan sound type.

## Implementation Steps
### 1. **Feature Extraction**
- Audio files are loaded using `librosa.load()`.
- MFCC features are extracted using `librosa.feature.mfcc()`.
- The mean of the MFCC features is computed to create a fixed-size feature vector.

### 2. **Data Preprocessing**
- Labels are encoded using `LabelEncoder`.
- Features and labels are converted to NumPy arrays.
- Labels are one-hot encoded using `np_utils.to_categorical()`.
- Data is split into training and testing sets (80-20 split).

### 3. **Neural Network Model**
- **Input Layer:** 40-dimensional MFCC feature vector.
- **Hidden Layers:**
  - Dense (256 neurons, ReLU activation, Dropout 0.3)
  - Dense (512 neurons, ReLU activation, Dropout 0.3)
  - Dense (256 neurons, ReLU activation, Dropout 0.3)
- **Output Layer:** Softmax activation for classification.
- **Loss Function:** Categorical Cross-Entropy.
- **Optimizer:** Adam.
- Model is trained for 100 epochs with a batch size of 32.

### 4. **Model Training & Evaluation**
- The model is trained with `model.fit()` and validated using a validation split of 20%.
- Accuracy is evaluated on the test set.
- The trained model is saved as `my_model.h5` for future predictions.

## Setup Instructions
1. **Clone this repository:**
   ```sh
   git clone https://github.com/your-repo/audio-classification.git
   cd audio-classification
   ```
2. **Install Dependencies:**
   ```sh
   pip install librosa pandas numpy tensorflow scikit-learn
   ```
3. **Run the Training Script:**
   ```sh
   python train_model.py
   ```



