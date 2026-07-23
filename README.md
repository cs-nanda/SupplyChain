# Supply Chain Demand Forecasting using Machine Learning

## Overview

This project develops a demand forecasting framework using the Walmart M5 Forecasting dataset to support inventory planning and supply chain decision-making.

The project compares multiple time-series forecasting techniques to identify the most effective model for predicting retail demand.

---

## Problem Statement

Accurate demand forecasting enables organizations to:

- Improve inventory planning
- Reduce stock-outs
- Minimize excess inventory
- Support production and replenishment planning

---

## Dataset

**Walmart M5 Forecasting Dataset**

- Retail sales data
- 30,490 SKU-store time series
- Multiple stores and product categories
- Daily sales spanning five years

---

## Technologies Used

- Python
- Pandas
- NumPy
- Statsmodels
- Matplotlib

---

## Forecasting Models

- Naive Seasonal
- Holt-Winters ETS
- MSTL + ETS

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Symmetric Mean Absolute Percentage Error (SMAPE)

---

## Repository Structure

```text
Project_Report.pdf
Project_Presentation.pdf
data/
images/
src/
```

---

## Contents

📄 Project Report

Detailed explanation of the methodology, implementation, model comparison, evaluation, and conclusions.

📊 Project Presentation

Summary of the project, key findings, and business insights.

📁 Data

Contains generated forecasting outputs.

🖼 Images

Contains important visualizations generated during analysis.

💻 Source Code

Python scripts used for preprocessing, forecasting, and evaluation.

---

## Key Result

Among all evaluated models, **MSTL + ETS** achieved the best forecasting performance and generated reliable demand forecasts for inventory planning and operational decision-making.
