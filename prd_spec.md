# Product Requirements Document (PRD)

## Online Fraud Detection System

**Version:** 1.0\
**Status:** Initial project specification\
**Product type:** Machine Learning Classification System\
**Primary objective:** Identify potentially fraudulent financial
transactions and help reduce fraud losses.

------------------------------------------------------------------------

## 1. Product Overview

The Online Fraud Detection System is a machine-learning-powered
application that classifies financial transactions as **legitimate** or
**potentially fraudulent**. It analyses transaction characteristics such
as transaction amount, transaction timing, and transaction frequency to
identify patterns that may indicate fraud.

The initial model candidates are **Random Forest** and **XGBoost**.
Model performance will be evaluated primarily using **precision** and
**recall**, with particular attention to the cost of missed fraud and
false alarms.

The system is intended as a decision-support tool. A high-risk
prediction should be reviewed or handled according to a defined
operational policy; the model should not automatically be treated as
proof of fraud.

## 2. Problem Statement

Financial institutions and online payment platforms process large
numbers of transactions. Manual review cannot efficiently inspect every
transaction, while simple fixed rules may fail to identify new or
complex fraud patterns.

The project addresses this problem by learning from labelled historical
transaction data and assigning each transaction a fraud-risk prediction.
This can help analysts prioritise suspicious activity and respond more
quickly.

## 3. Goals and Objectives

### Primary goals

-   Classify each eligible transaction as legitimate or potentially
    fraudulent.
-   Identify suspicious patterns using transaction amount, time, and
    frequency.
-   Compare Random Forest and XGBoost on the same data split and
    evaluation protocol.
-   Prioritise fraud detection while controlling unnecessary false
    alerts.
-   Present prediction results in a clear, understandable format.

### Success measures

-   Report precision, recall, F1-score, and confusion matrix on a
    held-out test set.
-   Track the false-positive rate and, where possible, precision-recall
    AUC.
-   Record inference latency for individual transactions or batches.
-   Document the chosen operating threshold and the trade-off it
    creates.

No performance percentage is promised in advance. Targets should be set
after inspecting the dataset, class balance, and baseline results.

## 4. Target Users

1.  **Fraud analyst:** Reviews high-risk transactions and investigates
    alerts.
2.  **Risk or operations manager:** Monitors fraud trends and model
    performance.
3.  **Developer / data scientist:** Prepares data, trains models,
    evaluates results, and maintains the system.
4.  **Demonstration user:** Runs sample transactions to understand the
    system's predictions.

## 5. Scope

### In scope (MVP)

-   Load a labelled Kaggle fraud-detection dataset.
-   Inspect, clean, and prepare transaction data.
-   Explore class imbalance and feature distributions.
-   Engineer features for transaction amount, transaction time, and
    transaction frequency where the dataset supports them.
-   Train and compare Random Forest and XGBoost classifiers.
-   Evaluate models using precision, recall, F1-score, confusion matrix,
    and precision-recall AUC where appropriate.
-   Select a decision threshold based on validation results.
-   Accept a transaction or batch of transactions and return a
    fraud-risk score and predicted class.
-   Display a simple prediction summary and model evaluation results.
-   Document limitations, assumptions, and reproducibility steps.

### Out of scope (initial release)

-   Direct connection to live banking or payment networks.
-   Automatic blocking, reversal, or cancellation of real transactions.
-   Guaranteed detection of all fraud.
-   Identity verification, KYC, AML case management, or law-enforcement
    workflows.
-   Continuous online learning or automatic model retraining.
-   Production-grade regulatory certification.

## 6. Dataset and Data Requirements

**Source:** Kaggle --- Fraud Detection Dataset (the exact Kaggle dataset
URL and version must be recorded before implementation).

The phrase "Fraud Detection Dataset" can refer to several different
Kaggle datasets. Before development, confirm the selected dataset, its
licence, schema, target column, and whether it is synthetic or based on
real transactions.

### Expected data fields

  -----------------------------------------------------------------------
  Feature                 Description             Handling
  ----------------------- ----------------------- -----------------------
  Transaction amount      Monetary value of a     Numeric; inspect skew
                          transaction             and outliers

  Transaction time        Timestamp or elapsed    Extract hour, day, or
                          time field              time interval when
                                                  valid

  Transaction frequency   Number of transactions  Engineer using
                          by an                   time-ordered grouping
                          account/card/customer   if identifiers and
                          in a defined time       timestamps exist
                          window                  

  Fraud label             Ground-truth target     Classification target;
                          indicating fraudulent   never use as an input
                          or legitimate           feature
                          transaction             

  Transaction/account     Dataset-specific        Use for grouping or
  identifiers             identifiers             feature engineering
                                                  only when justified;
                                                  avoid accidental
                                                  leakage
  -----------------------------------------------------------------------

The actual dataset may use different column names or may not contain all
expected fields. Map the real schema to these concepts rather than
assuming the columns exist.

### Data preparation requirements

-   Remove exact duplicate records where appropriate and document the
    rule.
-   Investigate missing values, invalid timestamps, impossible amounts,
    and inconsistent labels.
-   Preserve the original target distribution in reporting.
-   Encode categorical fields using a method appropriate to the chosen
    model.
-   Fit preprocessing steps only on training data to prevent leakage.
-   Do not use information that would only become available after the
    transaction being scored.
-   Use a chronological split when timestamps are meaningful; otherwise
    use a stratified split and explain why.
-   Keep a final test set untouched until model selection and threshold
    tuning are complete.

### Class imbalance

Fraud datasets are often highly imbalanced. Accuracy alone can therefore
be misleading. Consider class weights or carefully applied resampling on
the training data only. Do not resample the validation or test sets.

## 7. Machine Learning Approach

### Candidate models

**Random Forest** - Ensemble of decision trees. - Captures nonlinear
relationships and feature interactions. - Provides a useful baseline and
feature-importance estimates. - Can be robust with limited feature
scaling requirements.

**XGBoost** - Gradient-boosted decision trees. - Can model complex
patterns and often performs well on tabular data. - Supports
class-imbalance weighting and regularisation. - Requires careful tuning
to reduce overfitting.

### Training workflow

1.  Load and validate the dataset.
2.  Perform exploratory data analysis.
3.  Define leakage-safe features and target.
4.  Create train, validation, and test partitions.
5.  Build preprocessing and model pipelines.
6.  Train a baseline model and both candidate models.
7.  Tune hyperparameters using training/validation data only.
8.  Compare models and select an operating threshold on validation data.
9.  Evaluate the selected model once on the held-out test set.
10. Save the model, preprocessing pipeline, configuration, and
    evaluation report.

### Prediction output

For each transaction, the system should return: - Predicted class:
`Legitimate` or `Potentially Fraudulent`. - Fraud-risk probability or
score, if the model supports a calibrated interpretation. - Selected
threshold used for classification. - Model/version identifier. - A clear
notice that the output is a risk estimate, not a confirmed fraud
finding.

## 8. Evaluation Metrics

  -----------------------------------------------------------------------
  Metric                  Meaning                 Why it matters
  ----------------------- ----------------------- -----------------------
  Precision               Of transactions flagged Helps control false
                          as fraud, the           alarms
                          proportion that are     
                          actually fraud          

  Recall                  Of all actual           Measures how much fraud
                          fraudulent              is caught
                          transactions, the       
                          proportion detected     

  F1-score                Harmonic mean of        Balances both metrics
                          precision and recall    

  Confusion matrix        Counts true positives,  Shows the types of
                          false positives, true   classification errors
                          negatives, and false    
                          negatives               

  PR-AUC                  Area under the          Useful for imbalanced
                          precision-recall curve  classification

  Inference latency       Time required to        Indicates operational
                          produce a prediction    responsiveness
  -----------------------------------------------------------------------

**Priority:** Precision and recall are the headline metrics. The
preferred balance depends on the cost of missed fraud versus the cost of
investigating legitimate transactions. Report the threshold alongside
the metrics.

## 9. Functional Requirements

-   **FR-01:** The system shall load the selected dataset from a
    documented local file or approved data source.
-   **FR-02:** The system shall validate required columns and report
    schema errors.
-   **FR-03:** The system shall display basic dataset statistics and
    fraud-class distribution.
-   **FR-04:** The system shall prepare features without using the fraud
    label as an input.
-   **FR-05:** The system shall train Random Forest and XGBoost models.
-   **FR-06:** The system shall evaluate both models using the agreed
    metrics.
-   **FR-07:** The system shall allow a user to submit a transaction for
    prediction.
-   **FR-08:** The system shall return a predicted class and risk score
    where available.
-   **FR-09:** The system shall display the selected model and
    classification threshold.
-   **FR-10:** The system shall provide a clear warning that predictions
    require appropriate review.

## 10. Non-Functional Requirements

-   **Usability:** Results and metrics should be understandable to a
    non-technical reviewer.
-   **Performance:** Prediction latency should be measured and
    documented; set a numeric target after selecting the deployment
    environment.
-   **Reliability:** Invalid or incomplete inputs should produce
    understandable validation errors.
-   **Security:** Do not expose sensitive transaction or customer data
    in logs or demonstrations.
-   **Privacy:** Use only data permitted by the dataset licence and
    applicable policies; minimise personal data.
-   **Reproducibility:** Record dataset version, random seeds, feature
    list, model parameters, and package versions.
-   **Maintainability:** Keep data preparation, training, evaluation,
    and inference code modular.

## 11. User Workflow

1.  User opens the fraud-detection application or notebook.
2.  User loads the selected dataset.
3.  System validates the schema and shows a dataset summary.
4.  Developer trains and evaluates the candidate models.
5.  User selects a transaction or enters supported feature values.
6.  System generates a risk score and classification.
7.  User reviews the result and follows the organisation's investigation
    process.

## 12. Proposed Technical Stack

  -----------------------------------------------------------------------
  Layer                               Suggested technology
  ----------------------------------- -----------------------------------
  Language                            Python

  Data processing                     pandas, NumPy

  Visualisation                       Matplotlib, Seaborn

  Machine learning                    scikit-learn, XGBoost

  Model persistence                   joblib or an equivalent safe
                                      model-artifact approach

  Interface (optional MVP)            Streamlit

  Dataset source                      Kaggle
  -----------------------------------------------------------------------

The stack is a proposal and may be adapted to the development
environment.

## 13. Risks and Mitigations

  -----------------------------------------------------------------------
  Risk                    Impact                  Mitigation
  ----------------------- ----------------------- -----------------------
  Severe class imbalance  Model may miss fraud    Use precision/recall,
                          while showing high      PR-AUC, class weights,
                          accuracy                and threshold tuning

  Data leakage            Unrealistically high    Use time-aware features
                          test performance        and splits; fit
                                                  transformations on
                                                  training data only

  Dataset does not        Required feature cannot Engineer frequency only
  contain frequency       be directly used        if identifiers and
  fields                                          timestamps support it;
                                                  otherwise document the
                                                  limitation

  Dataset bias or         Results may not         State dataset
  synthetic data          generalise to real      limitations and avoid
                          transactions            production claims

  False positives         Legitimate transactions Tune threshold and
                          may be flagged          route alerts for human
                                                  review

  Concept drift           Fraud patterns may      Monitor performance and
                          change over time        plan periodic
                                                  evaluation/retraining

  Sensitive information   Privacy and security    Minimise, protect, and
  exposure                harm                    redact sensitive data
  -----------------------------------------------------------------------

## 14. Acceptance Criteria

The MVP is acceptable when: - The exact Kaggle dataset, version,
licence, schema, and target label are documented. - Data preparation and
feature engineering are reproducible. - Train/validation/test separation
prevents leakage. - Both Random Forest and XGBoost have been trained and
evaluated. - Precision, recall, F1-score, confusion matrix, and the
selected threshold are reported. - The final test set is not used for
model tuning. - A sample transaction can be scored and its result
explained clearly. - Known limitations and the human-review requirement
are visible in the documentation.

## 15. Future Enhancements

-   Add transaction velocity features across multiple time windows.
-   Add model explainability using SHAP or feature-contribution
    summaries.
-   Add analyst feedback and a reviewed-alert workflow.
-   Monitor data drift and model performance over time.
-   Evaluate on additional datasets and more realistic temporal splits.
-   Add secure API integration and role-based access controls before any
    production deployment.

## 16. Deliverables

1.  `prd.md` --- this product requirements document.
2.  Dataset description and data dictionary.
3.  Exploratory data analysis notebook or report.
4.  Reproducible preprocessing and training pipeline.
5.  Random Forest and XGBoost model comparison.
6.  Evaluation report with precision, recall, F1-score, confusion
    matrix, and PR-AUC where applicable.
7.  Prediction interface or demonstration notebook.
8.  README containing setup, execution, and limitations.

------------------------------------------------------------------------

**Product principle:** The system flags risk to support faster
investigation. It does not independently prove that a transaction is
fraudulent or guarantee prevention of financial fraud.
