# Housing Price Prediction & Explainability

A Random Forest regression study examining how income, geographic location, and area-level housing characteristics relate to median housing values.

## Business value
Illustrate how analysts can explain a pricing model, distinguish average effects from individual variation, and examine correlated features.

## Technical strengths
Python, Pandas, scikit-learn, Matplotlib, Seaborn, and ALEPython.
- Median imputation and one-hot encoding.
- RandomForestRegressor with 100 trees and random_state=42.
- A single 80/20 training/test split.
- Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE), Accumulated Local Effects (ALE), and held-out permutation importance.

## Explore and run
Open [the notebook](housing_price_explainability.ipynb) in Google Colab, supply the housing CSV, replace the personal Drive path, and install the listed dependencies. The setup currently retains course-template paths; adjust those before execution.

## Data interpretation
The target is median_house_value. Income and location are important features in the saved analysis.
total_rooms and population describe areas, not individual homes. Confirm income units and all definitions against the original data source before quoting them.

## Limitations and next steps
- The current code does not perform hyperparameter search or cross-validation.
- Add test MAE, RMSE, R², and a simple regression baseline.
- Fit missing-value imputation on training data only.
- Add the original dataset citation, access instructions, and a pinned dependency file.
- Model effect plots show predictive associations, not causal effects or an individual-property appraisal.

## Project context
Developed by Yiren Shen for Duke's AIPI 590 Explainable AI course.
