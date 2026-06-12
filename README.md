# 🎯 Placement Readiness Analyzer

## Project Overview

Placement preparation is something most students focus on during their final years of study. However, it is often difficult to understand which factors contribute most to placement success and which areas need improvement.

I built this project to explore how machine learning can be used to analyze student profiles and estimate placement readiness based on academic performance, projects, internships, communication skills, aptitude, and extracurricular involvement.

The project not only predicts placement probability but also provides insights into strengths, skill gaps, and recommendations for improvement.

---

## Problem Statement

Students usually know their CGPA and project experience, but they often struggle to understand how different aspects of their profile contribute to placement opportunities.

The goal of this project is to:

* Analyze student placement-related data
* Identify important factors influencing placement
* Build machine learning models for prediction
* Provide personalized recommendations through a web application

---

## Dataset

Dataset Used:

`college_student_placement_dataset.csv`

**Note:**
The dataset used in this project is a simulated educational dataset intended for learning and demonstrating the complete machine learning workflow.

The objective was not to build a production-grade placement prediction system but to gain hands-on experience with:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning
* Model Deployment

---

## Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn

### Model Deployment

* Streamlit

### Model Persistence

* Joblib

---

## Project Workflow

### 1. Data Understanding

The dataset was first explored to understand:

* Number of records
* Data types
* Feature descriptions
* Target variable distribution

---

### 2. Data Cleaning

Performed:

* Duplicate checks
* Data type validation
* Category inspection
* Missing value verification

---

### 3. Exploratory Data Analysis (EDA)

Several visualizations were created to understand relationships between features and placement outcomes.

Examples:

* Placement Distribution
* CGPA vs Placement
* Communication Skills vs Placement
* Projects Completed vs Placement
* Internship Experience vs Placement

---

### 4. Feature Engineering

To improve predictive performance, additional features were created.

#### Technical Score

Combines:

* IQ
* Academic Performance
* Projects Completed

#### Readiness Score

Combines:

* CGPA
* Communication Skills
* Internship Experience
* Extracurricular Activities

#### Profile Strength

Overall score combining technical and readiness indicators.

Feature engineering helped transform raw data into more meaningful indicators.

---

## Machine Learning Models

### Logistic Regression

Used as the baseline model.

Results:

* Accuracy ≈ 90%

This model provided a benchmark for comparison.

---

### Random Forest Classifier

A Random Forest model was trained and evaluated.

Results:

* Near-perfect accuracy on the dataset

Since the dataset is synthetic and follows strong patterns, Random Forest was able to learn these relationships extremely well.

To investigate this behavior, multiple train-test splits and feature importance analysis were performed.

Important features included:

* Communication Skills
* IQ
* CGPA
* Projects Completed
* Profile Strength

---

## Streamlit Application

The project includes an interactive Streamlit application where users can:

* Enter profile details
* Analyze placement readiness
* View placement probability
* Identify strengths
* Discover improvement areas
* Receive recommendations

The application is designed as a simple placement readiness assessment tool rather than a final hiring decision system.

---

## Project Structure

```text
Student_Placement_Engine/

app/
│
└── careerlaunch_ai.py

data/
├── raw/
└── processed/

models/
└── random_forest_model.pkl

reports/
└── figures/

src/
├── data_understanding.py
├── data_cleaning.py
├── advanced_eda.py
├── feature_engineering.py
├── train_model.py
├── random_forest_model.py
├── save_model.py
└── predict.py

README.md
requirements.txt
```

---

## How to Run

### Clone Repository

```bash
git clone <repository-url>
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app/careerlaunch_ai.py
```

---

## Key Learnings

Through this project, I gained practical experience in:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Classification models
* Model evaluation
* Streamlit deployment
* End-to-end machine learning workflow

---

## Future Improvements

Potential future enhancements include:

* Using real placement reports from colleges
* Collecting industry placement data
* Improving UI/UX
* Adding explainable AI techniques
* Deploying the application online

---

## Author

Manju

This project was developed as part of my machine learning learning journey and portfolio development.
