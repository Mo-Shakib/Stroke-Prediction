# Stroke Prediction Project

This repository contains a Stroke Prediction project implemented in Python using machine learning techniques. The goal of this project is to predict the likelihood of a person having a stroke based on various demographic, lifestyle, and medical factors.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Stroke is a medical condition that occurs when the blood supply to the brain is interrupted or reduced, resulting in damage to brain cells. Early detection and prediction of stroke risk can help healthcare professionals take preventive measures and provide appropriate treatment.

This project aims to build a machine learning model that can predict the probability of stroke based on various attributes such as age, gender, hypertension, heart disease, smoking status, etc. The model is trained using a dataset consisting of these attributes and corresponding stroke labels.

## Dataset

The dataset used for this project contains the following features:

- `id`: Unique identifier for each individual
- `gender`: Male or Female
- `age`: Age of the individual in years
- `hypertension`: Whether the individual has hypertension (0 - No, 1 - Yes)
- `heart_disease`: Whether the individual has heart disease (0 - No, 1 - Yes)
- `ever_married`: Whether the individual is ever married (Yes or No)
- `work_type`: Type of work (Private, Self-employed, Govt_job, children, Never_worked)
- `Residence_type`: Type of residence (Urban or Rural)
- `avg_glucose_level`: Average glucose level in blood
- `bmi`: Body mass index
- `smoking_status`: Smoking status of the individual (formerly smoked, never smoked, smokes, unknown)
- `stroke`: Whether the individual had a stroke (0 - No, 1 - Yes)

The dataset used in this project is included in this repository. 

## Installation

1. Clone this repository to your local machine using the following command:

   ```
   git clone https://github.com/Mo-Shakib/Stroke-Prediction.git
   ```

2. Change to the project directory:

   ```
   cd Stroke-Prediction
   ```

3. Install the required dependencies. It is recommended to use a virtual environment:

   ```
   pip install -r requirements.txt
   ```

## Usage

The main script for running the stroke prediction model is `predict_stroke.py`. You can use this script to predict stroke probabilities for new data points. Here's an example of how to use it:

```python
from predictor import StrokePredictor

# Create an instance of the predictor
predictor = StrokePredictor()

# Load the pre-trained model
predictor.load_model('model.pkl')

# Make predictions for new data points
data = {
    'age': 45,
    'gender': 'Male',
    'hypertension': 0,
    'heart_disease': 0,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 80.5,
    'bmi': 25.5,
    'smoking_status': 'formerly smoked'
}

probability = predictor.predict(data)
print(f"Stroke probability: {probability}")
```

## Model Training

To train the stroke prediction model using the provided dataset, you can run the `train_model.py` script. This script loads the dataset, preprocesses the data, trains the model, and saves the trained model to a file.

```bash
python train_model.py
```

The trained model will be saved as `model.pkl` in the project directory.

## Evaluation

The performance of the trained model can be evaluated using the `evaluate_model.py` script. This script calculates various evaluation metrics such as accuracy, precision, recall, and F1-score.

```bash
python evaluate_model.py
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
