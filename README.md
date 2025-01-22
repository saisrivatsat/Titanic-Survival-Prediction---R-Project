# Titanic Survival Prediction Project

## Overview
This project uses predictive modeling to analyze Titanic passenger data and predict survival probabilities based on features such as class, gender, age, and family size. By combining data preprocessing, exploratory data analysis (EDA), and machine learning, the project builds and evaluates models to understand survival factors and generate actionable insights.

---

## Objectives
- **Research Question:** Can passenger survival on the Titanic be predicted based on various features?
- **Hypothesis:** Certain attributes (e.g., passenger class, gender, and age) influence survival probability.
- **Goal:** Build models with high accuracy to predict survival outcomes.

---

## Features
1. **Preprocessing:**
   - Handled missing values and outliers.
   - Removed columns with high cardinality for simplified analysis.
2. **Exploratory Data Analysis:**
   - Visualized distributions of features using histograms, box plots, and heatmaps.
   - Analyzed correlations between variables.
3. **Model Building:**
   - Logistic Regression: Predicts binary survival outcomes.
   - Random Forest: Combines decision trees for robust predictions.
4. **Model Evaluation:**
   - Metrics: Accuracy, Kappa, and Confusion Matrix.

---

## Files
- **Titanic_Code.R**: Contains the R script for data loading, cleaning, EDA, and model training.
- **Titanic_PPT.pdf**: Presentation summarizing the project workflow and findings.
- **Titanic_Report.pdf**: Comprehensive project report with detailed analysis and results.

---

## Dependencies
- **R Packages:**
  - `dplyr`: Data manipulation.
  - `ggplot2`: Visualization.
  - `caret`: Machine learning.
  - `ggcorrplot`: Correlation heatmaps.
  - `plotly`: Interactive plots.
  - `lubridate`: Date handling.
  - `readr`: CSV file reading.

---

## How to Run
1. **Clone Repository:**
   ```bash
   git clone <repository_url>
   ```
2. **Set Working Directory:**
   Ensure the Titanic dataset (`titanic.csv`) is available in the working directory.
3. **Install Dependencies:**
   Install required R packages using:
   ```R
   install.packages(c("dplyr", "ggplot2", "caret", "ggcorrplot", "plotly", "lubridate", "readr"))
   ```
4. **Run Script:**
   Execute the `Titanic_Code.R` script in RStudio or an equivalent IDE.

---

## Results
- **Logistic Regression Model:**
  - Accuracy: 80.65%
  - Kappa: 0.5912
- **Random Forest Model:**
  - Accuracy: 81.05%
  - Kappa: 0.5939

**Conclusion:** The Random Forest model slightly outperformed Logistic Regression, with better accuracy and Kappa metrics.

---

## Future Enhancements
- Incorporate advanced models like Gradient Boosting or Neural Networks.
- Engineer additional features (e.g., cabin location, lifeboat proximity).
- Explore external data sources for context (e.g., weather conditions).
