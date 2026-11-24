# 🧠 Brain Tumor Classification using Dual Attention DenseNet121

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Brain tumors are among the most critical neurological disorders, and accurate diagnosis plays a vital role in treatment planning. This project presents an AI-powered Brain Tumor Classification system that automatically classifies brain MRI images into four categories using a **Dual Attention DenseNet121** deep learning architecture.

The model is deployed through a Streamlit web application, allowing users to upload MRI images and receive instant predictions.

## 🎯 Objectives

- Detect brain tumors from MRI images.
- Classify MRI scans into four categories.
- Improve feature extraction using Dual Attention mechanisms.
- Build an easy-to-use web application for prediction.

## 🧬 Tumor Classes

- Glioma
- Meningioma
- Pituitary
- No Tumor

## 🚀 Features

- Deep Learning based MRI image classification
- DenseNet121 backbone architecture
- Dual Attention Module
  - Channel Attention
  - Spatial Attention
- Image preprocessing
- Streamlit Web Interface
- Fast prediction on uploaded MRI scans
- User-friendly interface

## 🏗️ Model Architecture

The proposed model consists of:

- Input Image (224 × 224 × 3)
- DenseNet121 Feature Extractor
- Channel Attention Module
- Spatial Attention Module
- Global Average Pooling
- Dense Layer
- Softmax Output Layer

The attention modules enable the network to focus on the most informative regions of MRI images, improving classification performance.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning |
| Keras | Model Development |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| OpenCV | Image Processing |
| Matplotlib | Visualization |
| Streamlit | Web Application |
| Google Colab | Model Training |
| Git & GitHub | Version Control |

## 📂 Dataset

The project uses a Brain MRI dataset containing four classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

Dataset Split:

- Training: 70%
- Validation: 15%
- Testing: 15%

## 📁 Project Structure

BrainTumorProject/
│
├── app.py
├── Untitled27.ipynb
├── baseline_densenet.h5
├── dual_attention_densenet.h5
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
```
## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/sheetala62/MCA_PROJECT.git
```
Move into the project folder

```bash
cd MCA_PROJECT
```

Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Performance

| Model | Test Accuracy |
|--------|--------------|
| Baseline DenseNet121 | 93.37% |
| Dual Attention DenseNet121 | 93.44% |

## 💡 Future Enhancements

- Improve classification accuracy using larger datasets
- Add Grad-CAM visualizations for explainability
- Deploy using Docker
- Cloud deployment (AWS/Azure)
- Mobile-friendly interface
- REST API integration

## 📷 Screenshots

Add screenshots of:

- Home Page
- Image Upload Interface
- Prediction Result
- Accuracy Graphs
- Training Curves

## 📚 Skills Demonstrated

- Deep Learning
- Computer Vision
- Medical Image Classification
- TensorFlow
- Keras
- Streamlit
- CNN
- DenseNet121
- Transfer Learning
- Attention Mechanisms
- Python
- Git & GitHub

## 👩‍💻 Author

Sheetala Hegde

🎓 Master of Computer Applications (MCA)

### Connect with me

- GitHub: https://github.com/sheetala62
- LinkedIn: *(Add your LinkedIn profile here)*

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

## 📄 License

This project is intended for educational and research purposes.
