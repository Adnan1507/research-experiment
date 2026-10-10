Methodology

1\. Research Question

Can a Random Forest model predict diabetes better than a Logistic Regression model using the Pima Indians Diabetes Dataset? I will compare both models using ROC-AUC, F1-score, and recall for diabetic patients after cleaning the hidden missing values in the dataset.

2\. Dataset

The dataset is the Pima Indians Diabetes Database, originally collected by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and available in the UCI repository.

The dataset contains 768 rows and 9 columns. It has 8 input features and 1 target column. There are no categorical columns.

The 8 input features are Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, and Age. The target column is Outcome, where 1 means the person has diabetes and 0 means the person does not have diabetes.

There are 500 people without diabetes and 268 people with diabetes. This means about 35% of the people in the dataset have diabetes.

Limitations of the dataset:

•	The dataset is small, so the results may change depending on how the data is divided into training and testing sets.

•	The two classes are not equal in size. There are more people without diabetes than people with diabetes.

•	The dataset only contains women aged 21 or older from the Pima heritage group. So, the results may not work equally well for other groups of people.

•	Some columns use 0 to represent missing values. For example, Glucose = 0 and BMI = 0 are not physically possible values. Insulin and SkinThickness have the highest number of missing values.

•	The data are old because they were collected before 1990.

3\. Data Cleaning Plan

First, I will check the dataset for duplicate rows and actual missing values, such as NaN values.

Next, I will treat 0 as a missing value in the Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns. I will replace these zeros with NaN. However, I will keep 0 in the Pregnancies column because it is a valid value.

Then, I will divide the dataset into two parts: 80% for training and 20% for testing. I will use stratified splitting so that both sets have a similar proportion of people with and without diabetes. I will also use a fixed random seed of 42 so that the same split can be reproduced.

I will split the data before filling in the missing values to prevent information from the test set from leaking into the training process.

After splitting, I will fill the missing values using the median of each column, calculated only from the training data. I will use the median because the data may be unevenly distributed and may contain outliers.

Finally, I will use box plots to check for outliers. I will keep the outliers unless they are clearly impossible values, because they may represent real patients.

4\. Feature Engineering Plan

I will make some changes to the existing features to help the models learn better.

First, I will add two new columns to show whether Insulin and SkinThickness values were missing. Each column will use 1 to mean the value was missing and 0 to mean it was not missing. This is because missing values themselves may contain useful information.

Next, I will create a new feature by multiplying Glucose by BMI. This will help me check whether the combination of these two features provides useful information for predicting diabetes.

I will also create age groups, such as 21–30, 31–40, 41–50, and 51+.

For Logistic Regression, I will use StandardScaler to scale the features. I will fit the scaler using only the training data to avoid data leakage. Random Forest does not need feature scaling.

Finally, I will check the importance of each feature and compare the results with previous research. I expect Glucose to be one of the most important features.

5\. Models and Why I Chose Them

Logistic Regression (Baseline Model):

Logistic Regression is a simple, fast, and easy-to-understand machine learning model. I will use it as the baseline model to compare its performance with Random Forest. Previous studies have also shown that Logistic Regression can perform well on this dataset.

Random Forest:

Random Forest combines multiple decision trees to make predictions. It can learn complex patterns, non-linear relationships, and interactions between features. It can also work with unscaled features and is generally less affected by outliers than some other models. Another benefit is that it can show which features are most important for making predictions.

I chose these two models because I want to compare a simple model with a more complex model. This will help me understand whether using a more complex model actually improves the results on this small dataset.

6\. Evaluation Metrics and Why I Chose Them

ROC-AUC (Main Metric):

ROC-AUC will be my main evaluation metric. It measures how well a model can separate people with diabetes from people without diabetes at different classification thresholds. It is useful when the classes are not equal in size.

Recall for the Diabetic Class:

Recall measures how many people who actually have diabetes are correctly identified by the model. I will focus on this metric because missing a person who has diabetes may be more serious than giving a false alarm.

F1-Score:

F1-score combines precision and recall into one value. It will help me understand how well the model balances correctly identifying diabetic patients with avoiding false positive predictions.

Accuracy:

Accuracy measures the percentage of all predictions that are correct. I will report it so that I can compare my results with earlier studies, which often use accuracy. However, it will not be my main metric because a model that always predicts "no diabetes" would already achieve about 65% accuracy on this dataset.

Confusion Matrix:

I will use a confusion matrix to see the different types of correct and incorrect predictions made by each model. It will show how many people with diabetes were correctly identified, how many were missed, and how many people without diabetes were incorrectly predicted to have diabetes.

Both models will be tested on the same held-out test set to ensure a fair comparison. Since the dataset is small, I will also use 5-fold cross-validation to calculate the ROC-AUC score. This will provide a more stable comparison of the two models' performance.





