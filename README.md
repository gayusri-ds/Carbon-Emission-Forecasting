# Carbon Emission Analysis and Forecasting

## Overview
This project encompasses data preprocessing, exploratory data analysis (EDA), and comparison of three different models for carbon emission prediction. The goal is to understand emission trends and identify the most effective predictive model. The dataset used for analysis is obtained from Kaggle and is stored in the file `emissions.csv`.

## Kaggle Dataset
The dataset used in this project is sourced from Kaggle and contains historical carbon emission data from various states, sectors, and fuel types. It provides valuable insights into emission trends and patterns, enabling analysis and prediction of future emissions. The dataset can be accessed [https://www.kaggle.com/datasets/alistairking/u-s-co2-emissions]

## Project Structure
The project is structured into several stages:

1. **Data Preprocessing**: Initial preprocessing steps include handling missing values, removing duplicates, and filtering data points.
2. **Exploratory Data Analysis (EDA)**: Exploratory analysis is conducted to uncover insights and trends in carbon emission data.
3. **Automated Exploratory Data Analysis (AutoEDA)**: A comprehensive report is generated using Sweetviz to analyze dataset characteristics.
4. **Outlier Detection and Treatment**: Outliers are detected and treated using the Winsorization method to ensure data integrity.
5. **Data Normality Testing**: Shapiro-Wilk tests are performed to assess the normality of data distributions.
6. **Stationarity Testing**: Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests are used to evaluate time series stationarity.
7. **Model Comparison**: Three different models, including LSTM, GRU, and SimpleRNN are trained and evaluated for carbon emission prediction.

## Dependencies
The project utilizes the following libraries:
- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `feature_engine`
- `sweetviz`
- `scipy`
- `statsmodels`
- `tensorflow`
- `scikit-learn`

## Usage
To run the project:
1. Ensure all required libraries are installed.
2. Download the `emissions.csv` dataset from Kaggle and place it in the project directory.
3. Execute the provided Python script to preprocess the data, conduct exploratory data analysis, and train different models for prediction.

## Files and Outputs
- `emissions.csv`: Input dataset containing carbon emission data.
- `My_Report.html`: Automated exploratory data analysis report generated using Sweetviz.
- `stationary_data_30.csv`: CSV file containing stationary time series data.
- `non_stationary_data_30.csv`: CSV file containing non-stationary time series data.
- **Trained models**: Models are saved in separate folders (`/lstm`, `/gru`, `/simple_rnn`) named according to the state, sector, and fuel type.

  
## Model Evaluation
The performance of each model was evaluated based on both training and test accuracy. The results are as follows:

| Algorithms | Train Accuracy | Test Accuracy |
|------------|----------------|---------------|
| LSTM       | 99.68%         | 98.40%        |
| SimpleRNN  | 98.55%         | 99.76%        |
| GRU        | 99.70%         | 98.64%        |

## Results and Insights
The project provides insights into carbon emission trends, identifies outliers, evaluates data normality and stationarity, and compares the performance of different forecasting models. The findings contribute to environmental analysis and can inform policy-making and sustainability efforts.

