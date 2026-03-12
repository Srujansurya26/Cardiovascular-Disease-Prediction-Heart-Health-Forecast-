Cardiovascular Disease Prediction Using Ensemble Technique

This project is a **Machine Learning based web application** that predicts cardiovascular diseases using **ECG images**.  
The system processes ECG images, extracts features, and applies **ensemble machine learning algorithms** to classify heart conditions.

The application is built using **Python, Scikit-learn, OpenCV, and Streamlit** for an interactive web interface.

---

## Features

- Upload ECG image for analysis
- ECG image preprocessing and feature extraction
- Ensemble machine learning model predictions
- Interactive web interface using Streamlit
- Visualization of uploaded ECG image and processed outputs

---

## Technologies Used

**Programming Language**
- Python

**Libraries & Frameworks**
- Streamlit
- NumPy
- OpenCV
- Scikit-learn
- Pillow

**Machine Learning Algorithms**
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Naive Bayes
- Logistic Regression

---

## Project Structure


Heart-disease-prediction-using-ensemble-technique
│
├── final_app.py
├── Ecg.py
├── Dockerfile
├── Contour_Leads_1-12_figure.png
├── CVPD_Final_PPT.pdf
├── CVPD_Final_PPT.pptx
│
├── training dataset ecg
│ ├── ECG Images of Myocardial Infarction Patients
│ ├── ECG Images of Patient that have History of MI
│ ├── ECG Images of Patient that have abnormal heartbeat
│ └── Normal Person ECG Images
│
└── pycache


---

## System Requirements

Make sure the following software is installed:

- Python **3.7 or higher**
- pip package manager
- Internet connection for installing dependencies

---

## Step 1: Clone or Download the Project

Download the project ZIP file and extract it.

Or clone the repository:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git

Navigate to the project folder:

cd Heart-disease-prediction-using-ensemble-technique
Step 2: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps dependencies isolated.

Windows
python -m venv venv
venv\Scripts\activate
Mac / Linux
python3 -m venv venv
source venv/bin/activate
Step 3: Install Required Dependencies

Install all required libraries using pip:

pip install streamlit numpy opencv-python scikit-learn pillow
Step 4: Run the Application

Start the Streamlit application:

streamlit run final_app.py
Step 5: Open the Application

Once the application starts, open your browser and go to:

http://localhost:8501
Step 6: Using the Application

Open the web interface.

Upload an ECG image (.jpg or .png).

The system will:

Display the uploaded ECG image

Perform preprocessing

Run ensemble machine learning models

Display prediction results.

Dataset

The dataset used in this project contains ECG images of different heart conditions:

Normal ECG images

Myocardial Infarction ECG images

Abnormal heartbeat ECG images

Patients with history of myocardial infarction

These datasets are used to train and evaluate the machine learning models.

Docker Support (Optional)

You can also run the application using Docker.

Build Docker image:

docker build -t heart-disease-predictor .

Run Docker container:

docker run -p 8501:8501 heart-disease-predictor

Open the application:

http://localhost:8501
Future Improvements

Improve model accuracy using deep learning techniques

Deploy the application on cloud platforms

Add patient health data input

Improve ECG signal segmentation and feature extraction

Author

Srujan Surya
Computer Science Engineering Student
Machine Learning & Full Stack Developer

LinkedIn: https://www.linkedin.com/in/srujan-surya-494439287
