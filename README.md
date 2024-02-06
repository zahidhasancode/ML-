**GDP Calculation Using Polynomial Regression**

This document outlines the process of analyzing and predicting the Gross Domestic Product (GDP) of India using polynomial regression. The analysis is based on historical data spanning several decades, focusing on GDP in billion USD, per capita income in USD, and percentage growth year-over-year.

**Setup**

The analysis begins with importing necessary libraries for data manipulation, visualization, and machine learning:

**Pandas and NumPy for data handling.**

Seaborn and Matplotlib for data visualization.
Scikit-learn for machine learning models and metrics.
**python**

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

**Data Overview**

The dataset, India_GDP_Data.csv, includes columns for the year, GDP in billion USD, per capita income in USD, and annual percentage growth. Initial exploration with .head() and .info() methods provides insight into the structure and types of data available.

**Visualization**

Correlation between variables is visualized using a heatmap, and relationships between the year and GDP, as well as the year and per capita income, are explored through scatter plots and regression plots.

**Data Preprocessing**

Unnecessary features, such as percentage growth and per capita income, are dropped to focus the analysis on the relationship between the year and GDP in billion USD.

**Feature and Target Variables**

The year is used as the independent variable (X), and GDP in billion USD is the dependent variable (y), setting the stage for polynomial regression.

**Polynomial Regression Model**

A polynomial regression model is trained to fit the historical GDP data, allowing for predictions that account for non-linear trends observed over the years. The model's performance is evaluated using standard metrics: mean squared error (MSE), mean absolute error (MAE), and the R-squared (R2) score.

**Future GDP Prediction**

The trained model is then used to predict future GDP values, extending the analysis to the year 2050. Predictions are visualized alongside historical data to assess the model's extrapolation capabilities.

**Conclusion**

This analysis demonstrates the application of polynomial regression to economic data, providing insights into past trends and future expectations for India's GDP. The approach highlights the usefulness of machine learning techniques in understanding complex, non-linear relationships in time-series data.
