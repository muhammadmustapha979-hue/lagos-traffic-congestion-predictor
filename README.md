# Lagos Traffic Congestion Predictor

**3MTT AI/ML Project | Brief AI-16**

## 1. Project Overview

This project develops a supervised machine-learning classifier that predicts traffic congestion as **Low, Medium, or High** using historical smart-mobility observations.

The project is designed around the 3MTT problem statement: Lagos commuters need route congestion estimates.

### Important data limitation

The selected public dataset is a general smart-mobility dataset and is **not official Lagos traffic data**. Its coordinates are outside Lagos. Therefore, this project is presented as a **proof-of-concept methodology contextualised for the Lagos use case**, not as a validated Lagos traffic forecasting system.

Official Lagos traffic data should be used before deployment.

## 2. Dataset

Source: Smart Mobility Traffic Dataset on Kaggle.

The dataset contains 5,000 observations and 15 original columns, including:

- Timestamp
- Latitude / Longitude
- Vehicle Count
- Traffic Speed
- Road Occupancy
- Traffic Light State
- Weather Condition
- Accident Report
- Sentiment Score
- Ride-Sharing Demand
- Parking Availability
- Emission Levels
- Energy Consumption
- Traffic Condition (target)

The original data contain no missing values and no duplicate records.

## 3. Machine Learning Task

This is a **multi-class classification** problem.

Target classes:

- Low
- Medium
- High

## 4. Workflow

1. Load and inspect the data.
2. Validate data quality.
3. Convert timestamp information.
4. Engineer hour, day-of-week, weekend and peak-period features.
5. Perform exploratory data analysis.
6. Investigate possible target leakage / label construction.
7. Split the data using a stratified train/test split.
8. Preprocess numerical and categorical variables using a scikit-learn pipeline.
9. Compare Logistic Regression, Decision Tree and Random Forest.
10. Evaluate using accuracy, macro precision, macro recall, macro F1 and a confusion matrix.
11. Save the final Random Forest pipeline with joblib.

## 5. Why Random Forest?

Random Forest was selected as the final model because it handles nonlinear relationships and mixed traffic features well while remaining practical to train and explain for this project.

## 6. Important Evaluation Note

The target variable shows strong rule-like relationships with direct traffic measurements such as vehicle count, road occupancy and speed. A shallow decision tree can reproduce the target with high accuracy.

This suggests that the supplied target may be derived from traffic measurements. Therefore, very high model accuracy should **not** be interpreted as evidence of real-world Lagos prediction performance.

This limitation is documented intentionally for methodological transparency.

## 7. Repository Structure

```text
lagos_traffic_congestion_predictor/
│
├── data/
│   └── smart_mobility_dataset.csv
│
├── figures/
│   ├── traffic_condition_distribution.png
│   ├── traffic_by_hour.png
│   ├── vehicle_count_vs_speed.png
│   ├── occupancy_vs_speed.png
│   └── confusion_matrix.png
│
├── models/
│   └── traffic_congestion_model.joblib
│
├── traffic_congestion_predictor.ipynb
├── predict.py
├── model_results.csv
├── evaluation_report.txt
└── README.md
```

## 8. Installation

```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
```

## 9. Running the Project

Open `traffic_congestion_predictor.ipynb` in Google Colab or Jupyter Notebook and run the cells from top to bottom.

## 10. Future Work

For a production Lagos system:

- Acquire official Lagos traffic data.
- Add route/road identifiers.
- Add traffic-flow and travel-time measurements.
- Integrate live traffic APIs or sensors.
- Retrain and validate specifically on Lagos routes.
- Monitor model drift over time.

## 11. Project Ethics and Data Transparency

This project does not claim to possess official Lagos traffic records. The Lagos context comes from the 3MTT project brief, while the model training data come from a publicly available smart-mobility dataset.
