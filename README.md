# AI-Powered Preliminary Diagnosis Assistant for Viral STIs

## Project Overview

This project is a web-based preliminary diagnosis assistant developed
for educational and academic purposes.

The system provides a preliminary assessment based on user-provided
symptoms and selected health-related information. It focuses on four
viral sexually transmitted infections (STIs):

- HIV
- HPV
- HSV (Herpes)
- Hepatitis B

The system is designed to provide preliminary assessment support and
does not replace professional medical diagnosis or laboratory testing.

## Key Features

- User-friendly web interface
- Symptom-based preliminary assessment
- Natural language symptom processing
- Machine learning-based prediction
- HPV assessment functionality
- Preliminary result presentation
- Separate assessment and result pages
- MongoDB database integration

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Machine Learning
- Natural Language Processing (NLP)
- spaCy
- MongoDB
- Pandas
- NumPy
- Joblib

## Machine Learning

The application uses trained machine learning models to generate
preliminary assessment results from user-provided information.

The project includes trained model files for the implemented
assessment functionality.

## Natural Language Processing

Natural language processing is implemented using spaCy to process
user-provided symptom descriptions and identify relevant symptom
information for the preliminary assessment process.

## Project Structure

```text
AI-STI-Preliminary-Diagnosis-Assistant/
│
├── models/
│   ├── hpv_biopsy_features.pkl
│   ├── hpv_biopsy_imputer.pkl
│   ├── hpv_biopsy_random_forest.pkl
│   ├── hpv_biopsy_threshold.pkl
│   ├── viral_sti_features.pkl
│   └── viral_sti_random_forest.pkl
│
├── app.py
├── home.html
├── hpv_assessment.html
├── hpv_result.html
├── result.html
├── symptom_checker.html
├── style.css
├── requirements.txt
└── README.md
