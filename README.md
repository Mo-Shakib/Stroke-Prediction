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

The [dataset](dataset.csv) used for this project contains the following features:

- `id`: The unique identifier for each individual.
- `gender`: The gender of the individual (Male or Female).
- `age`: The age of the individual in years.
- `hypertension`: Indicates whether the individual has hypertension (0 for No, 1 for Yes).
- `heart_disease`: Indicates whether the individual has a heart disease (0 for No, 1 for Yes).
- `ever_married`: Indicates whether the individual is ever married (No or Yes).
- `work_type`: The type of work the individual is engaged in (children, government, private, self-employed, or never worked).
- `Residence_type`: The type of residence of the individual (Rural or Urban).
- `avg_glucose_level`: The average glucose level in the individual's blood.
- `bmi`: The body mass index (BMI) of the individual.
- `smoking_status`: The smoking status of the individual (formerly smoked, never smoked, or smokes).
- `stroke`: Indicates whether the individual had a stroke (0 for No, 1 for Yes).

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
# Make predictions for new data points
user_data = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': ever_married,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'female': gender == "female",
    'male': gender == "male",
    'Other': gender == "other",
    'government_work': work_type == "government",
    'private_work': work_type == "private",
    'self_employed': work_type == "self-employed",
    'children_work': work_type == "children",
    'never_worked': work_type == "never worked",
    'rural_resident': residence_type == "rural",
    'urban_resident': residence_type == "urban",
    'formerly_smoked': smoking_status == "formerly smoked",
    'never_smoked': smoking_status == "never smoked",
    'smokes': smoking_status == "smokes"
}

with open('model_name.pkl', 'rb') as file:
    predictor = pickle.load(file)

X_test = np.array([list(user_data.values())])
predictions = predictor.predict(X_test)

```

## Model Training

To train the stroke prediction model using the provided dataset, you can run the `train_model.py` script. This script loads the dataset, preprocesses the data, trains the model, and saves the trained model to a file.

```bash
python train_model.py
```

The trained models will be saved as `RandomForest.pkl, LinearSVC.pkl, NeuralNetwork.pkl, LogisticRegression.pkl, KNN.pkl` in the project directory.

## Prediction

The outcome of the trained model can be tested using the `predict_stroke.py` script. Providing user input, one can predict the outcome.

```bash
python predict_stroke.py
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
