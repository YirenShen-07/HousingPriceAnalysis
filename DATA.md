# California census housing data

The maintenance run downloads the 20,640-row housing.csv distributed in [Aurélien Géron's Hands-On Machine Learning repository](https://github.com/ageron/handson-ml2/tree/master/datasets/housing). The original coursework cited [Shibu Mohapatra's California House Price mirror](https://www.kaggle.com/datasets/shibumohapatra/house-price). The run records the exact downloaded fingerprint in reports/run.json.

| Field | Meaning |
| --- | --- |
| longitude / latitude | District location coordinates |
| housing_median_age | Median district housing age in years, capped |
| total_rooms / total_bedrooms | District-wide totals, not individual-home counts |
| population / households | District totals |
| median_income | District median income, scaled in tens of thousands of dollars and capped |
| ocean_proximity | Categorical coast proximity |
| median_house_value | District median housing value in dollars, capped |

This is historic California census data commonly associated with the 1990 census. It is not a current real-estate dataset. No causal effect of income or location is established. Downloaded source data is cached locally, not redistributed with a new license.

Methods: [PDP / ICE](https://scikit-learn.org/stable/modules/partial_dependence.html), [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), Apley & Zhu, [Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models](https://arxiv.org/abs/1612.08468). The compact one-dimensional ALE implementation uses quantile bins, finite differences and observation-weight centering.
