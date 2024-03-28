# Student Score Prediction System

An end-to-end machine learning project that predicts student math scores based on various features. This project demonstrates a complete ML pipeline with data ingestion, transformation, model training, evaluation, and a web-based prediction interface.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Project Pipeline](#project-pipeline)
- [Web Application](#web-application)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Author](#author)

## 🎯 Overview

This project implements a complete machine learning workflow to predict student math scores based on demographic and educational factors. The system includes:

- **Data Processing Pipeline**: Automated data ingestion, cleaning, and transformation
- **Multiple Model Training**: Evaluates 7 different regression algorithms
- **Hyperparameter Tuning**: Uses GridSearchCV for optimal parameter selection
- **Model Evaluation**: R² score-based model comparison
- **Web Interface**: Flask-based application for real-time predictions
- **Production-Ready Code**: Comprehensive error handling, logging, and model persistence

## ✨ Features

- **Automated Data Pipeline**: End-to-end data processing workflow
- **Multiple ML Algorithms**: 
  - Random Forest Regressor
  - Decision Tree Regressor
  - Gradient Boosting Regressor
  - Linear Regression
  - XGBoost Regressor
  - CatBoost Regressor
  - AdaBoost Regressor
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Model Persistence**: Save and load trained models using dill
- **REST API**: Flask-based backend for predictions
- **Web UI**: HTML templates for user-friendly predictions
- **Comprehensive Logging**: Detailed execution logs for debugging
- **Custom Exception Handling**: Project-specific error management

## 📁 Project Structure

```
Student_Score_Predictor_Project/
├── app.py                          # Flask web application
├── main.py                         # Entry point for training pipeline
├── setup.py                        # Package setup file
├── requirements.txt                # Project dependencies
├── README.md                       # This file
├── src/
│   ├── __init__.py
│   ├── exception.py               # Custom exception classes
│   ├── logger.py                  # Logging configuration
│   ├── utils.py                   # Utility functions
│   ├── components/
│   │   ├── data_ingestion.py      # Data loading and train-test split
│   │   ├── data_transformation.py # Feature engineering and scaling
│   │   └── model_trainer.py       # Model training and evaluation
│   └── pipeline/
│       ├── train_pipeline.py      # Training orchestration
│       └── predict_pipeline.py    # Prediction workflow
├── notebooks/
│   ├── EDA STUDENT PERFORMANCE.ipynb    # Exploratory data analysis
│   └── 2. MODEL TRAINING.ipynb          # Model training notebook
├── templates/
│   ├── index.html                 # Home page
│   └── home.html                  # Prediction form
├── artifacts/
│   ├── train.csv                  # Training dataset
│   ├── test.csv                   # Test dataset
│   ├── data.csv                   # Raw dataset
│   ├── model.pkl                  # Trained model
│   └── preprocessor.pkl           # Data preprocessor
└── logs/                          # Application logs
```

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Student_Score_Predictor_Project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**
   ```bash
   pip install -e .
   ```

## 🚀 Usage

### Training the Model

Run the training pipeline to prepare data, train multiple models, and select the best performer:

```bash
python main.py
```

This will:
- Load and split the student dataset
- Transform features (encoding, scaling)
- Train 7 different regression models
- Perform hyperparameter tuning
- Evaluate and save the best model
- Generate logs in the `logs/` directory

### Running the Web Application

Launch the Flask web server for interactive predictions:

```bash
python app.py
```

The application will start at `http://localhost:5000`

**Features:**
- **Home Page** (`/`): Welcome page with links to prediction
- **Prediction Page** (`/predictdata`): Form to input student features and get predictions

### Making Predictions Programmatically

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create custom data
data = CustomData(
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education="bachelor's degree",
    lunch='standard',
    test_preparation_course='none',
    reading_score=72,
    writing_score=74
)

# Get prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(data.get_data_as_data_frame())
print(f"Predicted Math Score: {prediction[0]}")
```

## 📊 Dataset

The project uses a student performance dataset with the following features:

### Input Features
- **gender**: Student gender (male/female)
- **race_ethnicity**: Ethnicity group (A-E)
- **parental_level_of_education**: Parent's education level
- **lunch**: Type of lunch program
- **test_preparation_course**: Test prep completion status
- **reading_score**: Reading ability score (0-100)
- **writing_score**: Writing ability score (0-100)

### Target Variable
- **math_score**: Math performance score (0-100)

### Dataset Statistics
- **Total Samples**: 1000
- **Training Set**: 800 samples (80%)
- **Test Set**: 200 samples (20%)
- **No Missing Values**: Clean dataset

## 🤖 Models

### Implemented Algorithms

The project evaluates the following regression models:

1. **Linear Regression**
   - Baseline model for comparison
   - Fast training and inference

2. **Decision Tree Regressor**
   - Captures non-linear relationships
   - Hyperparameter tuning on criteria and max depth

3. **Random Forest Regressor**
   - Ensemble method with multiple trees
   - Robust to outliers

4. **Gradient Boosting Regressor**
   - Sequential tree building
   - Excellent predictive power

5. **XGBoost Regressor**
   - Optimized gradient boosting
   - Fast and memory-efficient

6. **CatBoost Regressor**
   - Handles categorical features naturally
   - Fast GPU support available

7. **AdaBoost Regressor**
   - Sequential error correction
   - Reduces bias through boosting

### Model Selection
- **Training Strategy**: GridSearchCV with 3-fold cross-validation
- **Evaluation Metric**: R² Score (coefficient of determination)
- **Best Model Selection**: Highest R² score on test set

## 🔄 Project Pipeline

### 1. Data Ingestion (`data_ingestion.py`)
- Reads raw data from CSV
- Performs train-test split (80-20)
- Saves split data to artifacts folder

### 2. Data Transformation (`data_transformation.py`)
- Encodes categorical variables
- Scales numerical features using StandardScaler
- Creates preprocessing pipeline
- Saves preprocessor for inference

### 3. Model Trainer (`model_trainer.py`)
- Initializes 7 regression models
- Defines hyperparameter search space
- Trains models with GridSearchCV
- Evaluates on test set
- Saves best model

### 4. Prediction Pipeline (`predict_pipeline.py`)
- Loads trained model and preprocessor
- Transforms new data
- Generates predictions

## 🌐 Web Application

The Flask application provides an interactive interface for predictions.

### Routes

- **GET `/`**: Home page
- **GET/POST `/predictdata`**: Prediction form and results

### Features
- HTML form for feature input
- Real-time predictions
- Result display
- Error handling

### Running the App

```bash
python app.py
```

Access the application at `http://0.0.0.0:5000`

## ⚙️ Configuration

### Dependencies

See [requirements.txt](requirements.txt) for all dependencies:

- **Data Processing**: pandas, numpy
- **ML Models**: scikit-learn, xgboost, catboost
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask
- **Serialization**: dill

### Package Configuration

The project is configured using `setup.py`:

```
Package Name: Machine_Learning_Project
Version: 0.0.1
Author: Girdhar
```

Install as editable package:
```bash
pip install -e .
```

## 📝 Logging

The project includes comprehensive logging:

- **Log Location**: `logs/` directory
- **Log Level**: INFO (info, warnings, errors)
- **Log Contents**: 
  - Data processing steps
  - Model training progress
  - Prediction requests
  - Error traces

## 🔍 Custom Exception Handling

The `exception.py` module provides custom exception handling with detailed error messages and stack traces for debugging.

## 📈 Performance

The trained models achieve strong R² scores on the test set. Model performance compared:
- Best performing models typically achieve R² > 0.85
- All models evaluated on same test set
- Hyperparameter tuning improves baseline performance

## 🚀 Future Enhancements

- [ ] Database integration for data persistence
- [ ] API authentication and rate limiting
- [ ] Model versioning and experiment tracking
- [ ] Cross-validation on entire dataset
- [ ] Feature importance analysis
- [ ] Model interpretability (SHAP values)
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Unit tests
- [ ] Performance monitoring dashboard

## 🛠️ Troubleshooting

### Issue: Model file not found
- **Solution**: Run training pipeline first with `python main.py`

### Issue: Import errors
- **Solution**: Install package with `pip install -e .`

### Issue: Port already in use
- **Solution**: Change Flask port in `app.py` or kill process using port 5000

### Issue: Data file not found
- **Solution**: Ensure `notebook/data/stud.csv` exists or update path in `data_ingestion.py`

## 📚 Project Notebooks

1. **EDA STUDENT PERFORMANCE.ipynb**
   - Exploratory data analysis
   - Statistical summaries
   - Visualization of relationships

2. **2. MODEL TRAINING.ipynb**
   - Model training walkthrough
   - Performance comparison
   - Hyperparameter optimization

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open-source and available under the MIT License.

## 👤 Author

**Girdhar**
- Email: girdharscs@gmail.com

## 🙏 Acknowledgments

- Student performance dataset
- Open-source machine learning community
- scikit-learn, XGBoost, and CatBoost documentation

---
**Status**: Active Development
