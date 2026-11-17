# Marketing Campaign Response Prediction & Customer Targeting

A machine-learning portfolio project that predicts whether a bank customer will subscribe to a term deposit after a direct-marketing campaign. It combines exploratory data analysis, feature engineering, preprocessing, classification, model evaluation and business interpretation.

## Project overview

| Item | Details |
|---|---|
| Problem | Binary classification |
| Domain | Banking / Marketing Analytics / Customer Targeting |
| Dataset | UCI Bank Marketing (`bank-additional-full.csv`) |
| Target | `y`, renamed to `target` (`yes` = 1, `no` = 0) |
| Models | Logistic Regression and Random Forest |
| Metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Tools | Python, Pandas, NumPy, Matplotlib, scikit-learn, Jupyter Notebook |

## Business problem

Banks use direct-marketing campaigns to encourage customers to subscribe to term deposits. Calling every customer consumes time and resources. A classification model can help estimate which customers are more likely to subscribe and help campaign teams prioritise outreach. The model is a decision-support tool; the best decision threshold depends on contact costs, campaign capacity and the value of a successful subscription.

## Dataset

The supplied `bank-additional-full.csv` contains **41,188 rows and 21 columns**. The raw target distribution is 36,548 `no` and 4,640 `yes` (approximately 88.7% vs. 11.3%), so the outcome is imbalanced. The CSV is semicolon-separated. The notebook removes 12 exact duplicate rows, leaving **41,176 rows**.

Variables cover customer demographics, contact and campaign history, economic indicators and the subscription outcome.

## Important modelling decisions

- **Avoiding data leakage:** `duration` is excluded because call duration is only known after the call. It would not be available when deciding whom to contact before a call.
- **Handling `pdays`:** the value `999` represents no previous contact in an earlier campaign. The notebook creates `previously_contacted` and removes `pdays`, avoiding treating 999 as an ordinary number of days.
- **Unknown categories:** recorded `unknown` values are retained as categories rather than automatically treated as missing values.
- **Train/test separation:** preprocessing is fitted on training data and applied to test data through scikit-learn pipelines.

## Workflow

1. Load and inspect the data (shape, columns, types, missing values and duplicates).
2. Remove exact duplicates and encode the target as 0/1.
3. Explore response distribution and subscription rates by age group, job, education and month.
4. Create `age_group` and `previously_contacted` features.
5. Exclude `duration` and replace `pdays` with the contact-history indicator.
6. Split into 80% training and 20% test sets using stratification (`random_state=42`).
7. Preprocess numeric and categorical features.
8. Train Logistic Regression and Random Forest using balanced class weights.
9. Compare metrics, confusion matrices, classification reports, ROC curves and Random Forest feature importance.
10. Translate patterns into campaign-planning considerations.

## Exploratory data analysis

### Campaign response distribution

The positive subscription class is the minority class.

![Distribution of campaign response](images/figure_10.png)

### Subscription rate by age group

![Subscription rate by age group](images/figure_12.png)

### Subscription rate by job category

![Subscription rate by job category](images/figure_14.png)

### Subscription rate by education

![Subscription rate by education](images/figure_16.png)

### Subscription rate by campaign month

![Subscription rate by month](images/figure_18.png)

**Selected descriptive observations from the notebook:**
- The `60+` age group has the highest observed subscription rate among the defined age bands (**45.54%**).
- `student` (**31.43%**) and `retired` (**25.26%**) are the highest-response job categories in the displayed top-five comparison.
- March (**50.55%**), December (**48.90%**), September (**44.91%**) and October (**43.93%**) show the highest observed monthly rates.

These are associations in this historical dataset, not evidence that these attributes cause subscription.

## Modelling and preprocessing

Numeric columns are median-imputed and standardised. Categorical columns are imputed with the most frequent category and one-hot encoded. The transformations are placed in a `ColumnTransformer` and fitted on the training set.

**Logistic Regression:** baseline linear classifier, `class_weight="balanced"`, `max_iter=1000`.

**Random Forest:** 150 trees, `max_depth=12`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight="balanced"`.

## Model results

Held-out test-set results from the notebook (rounded to three decimal places):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.830 | 0.358 | **0.642** | 0.460 | 0.800 |
| Random Forest | **0.861** | **0.423** | 0.633 | **0.507** | **0.812** |

Random Forest performs better on accuracy, precision, F1-score and ROC-AUC in this run. Logistic Regression has slightly higher recall. Since the target is imbalanced, accuracy alone is not sufficient for model selection.

- **Precision:** among customers predicted as subscribers, how many actually subscribed.
- **Recall:** among actual subscribers, how many the model identified.
- **F1-score:** harmonic balance between precision and recall.
- **ROC-AUC:** ranking/discrimination performance across thresholds.

### Confusion matrices

![Confusion matrix comparison](images/figure_30.png)

### ROC curves

![ROC curve comparison](images/figure_34.png)

### Random Forest feature importance

The feature-importance chart describes model behaviour; it does not prove that a feature causes subscription.

![Top Random Forest feature importances](images/figure_36.png)

## Business interpretation

- Use predicted probabilities to rank customers and prioritise a calling list.
- Balance the number of likely subscribers identified against the number of unsuccessful calls.
- Choose a decision threshold based on campaign capacity, contact cost and subscription value.
- Validate segment patterns on newer campaign data before using them operationally.
- Monitor performance because customer behaviour and economic conditions can change.

A higher-recall model may capture more actual subscribers but can also create more false positives. A higher-precision model may reduce unsuccessful calls but miss potential subscribers. The business objective should determine the trade-off.

## Repository structure

```text
Marketing-Campaign-Response-Prediction/
├── README.md
├── requirements.txt
├── Marketing_Campaign_Response_Prediction.ipynb
├── bank-additional-full.csv
└── images/
    ├── figure_10.png
    ├── figure_12.png
    ├── figure_14.png
    ├── figure_16.png
    ├── figure_18.png
    ├── figure_30.png
    ├── figure_34.png
    └── figure_36.png
```

## Run locally

1. Clone/download this repository and open its folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter:

```bash
jupyter notebook
```

4. Open `Marketing_Campaign_Response_Prediction.ipynb` and run cells from top to bottom. Keep `bank-additional-full.csv` beside the notebook. The notebook has been updated to use a portable relative file path.

## Limitations and future work

- Results are based on one stratified random train/test split; cross-validation and time-based validation would provide stronger evidence of generalisation.
- Metrics use the notebook's default prediction threshold. Threshold tuning should use validation data and an explicit business-cost framework.
- Historical customer behaviour and economic conditions may change.
- Feature importance and segment comparisons are not causal explanations.
- A real deployment would need monitoring, periodic validation/retraining, privacy and fairness review, and business approval of the operating threshold.

Possible extensions: probability calibration, cross-validation, threshold optimisation, lift/gains analysis, cost-sensitive evaluation and testing on newer campaign data.

## Dataset reference

This project uses the Bank Marketing dataset associated with the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). Retain the original dataset attribution and check its usage terms before redistribution.

## Disclaimer

Educational and portfolio project only. It is not a production banking decision system and should not be used for real customer decisions without further validation, governance and compliance review.
