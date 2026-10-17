\#Predicting Diabetes Onset on the Pima Indians Dataset: Logistic Regression versus Random Forest

\##Abstract

Early detection of diabetes can reduce serious complications, and machine learning offers a low-cost way to flag at-risk patients. This study asks whether a random forest predicts diabetes onset more accurately than a logistic regression baseline on the Pima Indians Diabetes dataset (768 women). Physiologically impossible zero values were treated as missing and filled with training-set medians, two missing-value indicators and two new features were added, and both models were evaluated on a stratified 20% test set and with five-fold cross-validation. The two models reached nearly identical test ROC-AUC (0.820 for random forest, 0.819 for logistic regression), and random forest was better on recall, F1 and accuracy. However, logistic regression scored higher in cross-validation (0.847 versus 0.824). Neither model caught more than 60% of diabetic cases. On this small dataset, the added complexity of a random forest gave no clear advantage.

\##1. Introduction

Diabetes is a major health problem, and catching it early gives patients a better chance of avoiding complications. The Pima Indians Diabetes dataset is a classic benchmark for this task. It comes from the National Institute of Diabetes and Digestive and Kidney Diseases and records clinical measurements for women from a population with a very high rate of diabetes.

The dataset was first used by Smith et al. (1988), who tested an early neural network called ADAP and found it reached about 76% sensitivity and specificity on held-out cases. Later work has compared many algorithms on the same data. Alzboon et al. (2025) tested eight classifiers and found the best accuracy was only about 78.6%, with random forest close behind at about 76.3%, so the differences between models were small. Wajahat and Keshtkar (2024) found that glucose was the strongest predictor in every model they tried, and that plain logistic regression outperformed random forest, k-nearest neighbours and a deep neural network. In contrast, Paul et al. (2025) reported accuracy above 88% for random forest and XGBoost, a figure far above the rest of the literature that is hard to evaluate because the abstract does not describe how the data were split or cleaned.

These mixed findings raise a practical question: does a more complex model really help on a small dataset? This paper therefore asks: \*Can a random forest classifier predict diabetes onset in the Pima Indians dataset more accurately than a logistic regression baseline, as measured by ROC-AUC, F1-score and recall on the diabetic class, after cleaning the dataset's hidden missing values?\*

\##2. Methodology

\###2.1 Dataset

The Pima Indians Diabetes Database contains 768 rows and 9 columns, all numeric. The eight features are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction and Age. The target is Outcome (1 = diabetes, 0 = no diabetes), with 500 negative and 268 positive cases (about 35% positive). All patients are women aged 21 or older of Pima heritage.

\### 2.2 Data cleaning

In Glucose, BloodPressure, SkinThickness, Insulin and BMI, a value of 0 is physiologically impossible and was treated as missing. The data had no duplicate rows. Pregnancies = 0 is valid and was kept. The data were then split into 80% training (614 rows) and 20% test (154 rows), stratified on Outcome with a fixed random seed of 42. Missing values were filled with the column median, calculated on the training set only, to prevent information from the test set leaking into training.

\### 2.3 Feature engineering

Before imputation, two indicator columns recorded whether Insulin and SkinThickness had been missing. Two new features were added: a Glucose × BMI interaction term and an age group (21-30, 31-40, 41-50, 51+). This gave 12 features in total. Features were standardized with StandardScaler (fit on the training set only) for logistic regression; random forest used the unscaled features.

\### 2.4 Models

Logistic regression was used as a simple, interpretable baseline, since earlier studies found it competitive on this dataset. A random forest (300 trees) was used as a more flexible model able to capture non-linear relationships and feature interactions. Comparing the two tests whether extra complexity helps.

\### 2.5 Evaluation

Both models were evaluated on the same held-out test set. The primary metric was ROC-AUC, which measures how well the model separates the classes across all thresholds and is not distorted by the class imbalance. Recall on the diabetic class was reported because missing a patient is more costly than a false alarm. F1, precision and accuracy were also reported, the last mainly to allow comparison with earlier studies. Because the test set is small, five-fold stratified cross-validated ROC-AUC on the training set was also computed.

\## 3. Results

\### 3.1 Test-set performance

\*\*Table 1. Performance on the held-out test set (154 rows) and cross-validated ROC-AUC (training set, 5 folds).\*\*

Model	ROC-AUC	Recall	Precision	F1	Accuracy	CV ROC-AUC (mean ± SD)

Logistic Regression	0.819	0.519	0.596	0.554	0.708	0.847 ± 0.019

Random Forest	0.820	0.593	0.653	0.621	0.747	0.824 ± 0.026



On the test set, the two models had almost the same ROC-AUC (0.820 for random forest and 0.819 for logistic regression). Random forest was better on recall (0.593 vs 0.519), precision, F1 (0.621 vs 0.554) and accuracy (0.747 vs 0.708), as shown in Figure 1. In cross-validation the order reversed: logistic regression scored 0.847 and random forest 0.824.

!\[Figure 1](figures/fig1\_metrics.png)

\### 3.2 Feature importance

Figure 2 shows which features the random forest relied on most. The top three features are Glucose\_BMI, Glucose, and BMI. Glucose is near the top, ranking second in individual feature importance, while the combined Glucose\_BMI feature ranks first. Because Glucose\_BMI is built from Glucose and BMI, the importance of glucose is partly shared between these features.

!\[Figure 2](figures/fig2\_importance.png)

\### 3.3 Types of errors

Figure 3 shows the confusion matrices. Of the 54 diabetic patients in the test set, logistic regression identified 28 and random forest identified 32, so both models missed a large share of real cases.

!\[Figure 3](figures/fig3\_confusion.png)

\## 4. Discussion

The main finding is that the more complex model did not clearly beat the simple one. The two models tie on the primary metric on the test set, random forest is somewhat better on the metrics that depend on a fixed 0.5 threshold, and logistic regression is better in cross-validation. This agrees with Wajahat and Keshtkar (2024), who also found logistic regression competitive, and with Alzboon et al. (2025), who found only small differences between models. It also suggests that on a small tabular dataset with a few informative features, a linear model can capture most of the available signal.

The accuracy of 71-75% is in line with the roughly 76-79% reported in earlier work and well below the 88% claimed by Paul et al. (2025). Without details of their data split we cannot tell why their figure is so high, but our results suggest that a value that high should be treated with caution.

The most important practical weakness is recall. Even the better model found only 32 of 54 diabetic patients (59%). In a screening setting this is the costly kind of error, so a real system would need a lower decision threshold or a different model.

\*\*Limitations.\*\*

•	The test set has only 154 rows (54 positive), so differences of a few points could come from the random split rather than from real differences between models.

•	All patients are women of Pima heritage aged 21 or older, so the results may not generalize to other populations.

•	The data were collected decades ago, and many Insulins and SkinThickness values were missing and had to be filled with medians, which may hide real patterns.

•	No hyperparameter tuning was done, and the decision threshold was left at 0.5.

•	Only two models were compared, and the results come from a single train/test split.

\## 5. Conclusion

This study compared logistic regression and random forest for predicting diabetes onset on the Pima Indians dataset after treating impossible zero values as missing and engineering a few extra features. The two models had almost identical test ROC-AUC (0.819 and 0.820), random forest was better on recall and F1, and logistic regression was better in cross-validation, so the extra complexity gave no clear benefit. Both models missed many diabetic patients, with recall of 52-59%. Future work should tune the models, lower the decision threshold to raise recall, try other algorithms such as gradient boosting, and validate on a larger and more diverse dataset.

\## References

Paul, R. U., Mbeledogu, N. N., Iduh, B. N., \& Okechukwu, O. P. (2025). A Comparative Analysis on diabetes using Ensemble Machine Learning Models on Pima Indian Datasets. Journal of Basic Physical Research, 14(2), 200–213.

Salem Alzboon, M., Al-Batah, M., Alqaraleh, M., Abuashour, A., \& Fuad Bader, A. (2025). A Comparative Study of Machine Learning Techniques for Early Prediction of Diabetes. arXiv. https://doi.org/10.48550/arXiv.2506.10180

Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., \& Johannes, R. S. (1988). Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Proceedings of the Annual Symposium on Computer Application in Medical Care, 261–265.

Wajahat, I., Keshtkar, F., \& Bukhari, S. A. C. (2024). Advancing Precision Healthcare Analytics: Machine Learning Approach-es for Diabetes Prognosis using the PIMA Indian Dataset. The International FLAIRS Conference Proceedings, 37. https://doi.org/10.32473/flairs.37.1.135329





