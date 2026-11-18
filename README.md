Bank Customer Churn Prediction and Retention Cost Optimization
A machine learning project that predicts bank customers who may leave the bank and estimates the potential financial impact of customer-retention campaigns.
2. Demo
Streamlit Application:http://13.238.219.180:8080
3. Problem Statement and Impact
Customer churn happens when customers stop using a bank's products or services. Losing customers can affect a bank's customer base and business growth.

The objective of this project is to build machine learning models that predict whether a customer is likely to leave the bank. The results can help banking teams:
- Identify customers who may be at risk of leaving.
- Understand customer behaviour associated with churn.
- Prioritise customers for retention campaigns.
- Estimate the possible cost and benefit of retention efforts.
This is an educational project. Predictions and financial estimates should be validated before being used for real banking decisions.
4. Dataset
- Dataset: Bank Customer Churn (BankChurners)
- Source: Kaggle dataset used in the notebook.
- Size: 10,127 rows and 23 columns.
- Target column: Attrition_Flag
  - Attrited Customer represents a customer who left.
  - Existing Customer represents a customer who stayed.
- Target distribution: 8,500 existing customers and 1,627 attrited customers.
The dataset includes customer demographics, banking relationship information, card usage, and transaction behaviour. Example features include Customer_Age, Gender, Education_Level, Months_Inactive_12_mon, Credit_Limit, Total_Revolving_Bal, Total_Trans_Amt, and Total_Trans_Ct.

The notebook reads the dataset from this relative path:
dataset/BankChurners.csv
5. Approach
1. Data loading and inspection: Load the CSV with Pandas and inspect shape, data types, duplicates, and target distribution.
2. Exploratory Data Analysis: Use charts and summary statistics to examine customer behaviour and churn patterns.
3. Data preparation: Remove the customer identifier (CLIENTNUM), the two Naive_Bayes_Classifier output columns, and Avg_Open_To_Buy because it repeats information represented by credit limit and revolving balance.
4. Target preparation: Encode Attrition_Flag as a binary target, with attrited customers represented as the positive class.
5. Feature engineering: Create Average_Transaction_Amount using transaction amount divided by transaction count.
6. Train-test split: Use an 80:20 stratified split with random_state=42 (8,101 training rows and 2,026 testing rows).
7. Outlier analysis: Detect potential outliers using the IQR method and boxplots. Retain the observations because extreme values may represent genuine customer behaviour.
8. Preprocessing: Use a Scikit-learn ColumnTransformer and pipelines. Numerical features are median-imputed and standardised; categorical features are imputed with Unknown and One-Hot Encoded.
9. Model building: Train Logistic Regression, Balanced Logistic Regression, Random Forest, Decision Tree, and Gradient Boosting models.
10. Evaluation and tuning: Compare accuracy, precision, recall, F1-score, and ROC-AUC. Use 5-fold Stratified Cross-Validation and GridSearchCV to tune Gradient Boosting.
11. Explainability: Use SHAP to understand which features influence the tuned model's predictions.
12. Business analysis: Estimate retention campaign costs and potential value using explicitly hypothetical assumptions.
6. Results
The following are the test-set results recorded in the notebook. Precision, recall, and F1-score refer to the positive class (Attrited Customer).
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.9126	0.8058	0.6000	0.6878	0.9318
Balanced Logistic Regression	0.8697	0.5634	0.8338	0.6725	0.9351
Random Forest	0.9536	0.9326	0.7662	0.8412	0.9874
Decision Tree	0.9319	0.7931	0.7785	0.7857	0.8698
Gradient Boosting	0.9635	0.9564	0.8092	0.8767	0.9886
Tuned Gradient Boosting	0.9684	0.9547	0.8431	0.8954	0.9921


The tuned Gradient Boosting model was selected as the final model. Its best parameters were:
- learning_rate: 0.1
- max_depth: 3
- n_estimators: 200
- subsample: 1.0
The best cross-validation F1-score reported by GridSearchCV was 0.9047. On the test set, the confusion matrix was [[1688, 13], [51, 274]].
7. Key Insights and Findings
- The dataset is imbalanced: 8,500 customers stayed and 1,627 left. Therefore, accuracy alone is not enough to evaluate churn detection; recall and F1-score are also important.
- Attrited customers had lower average transaction counts (44.9 compared with 68.7 for existing customers) and lower average transaction amounts (approximately 3,095 compared with 4,655).
- Attrited customers also had lower average revolving balances (673 compared with 1,257) and lower average utilisation ratios (0.16 compared with 0.30).
- Attrited customers contacted the bank more often on average (2.97 compared with 2.36) and used fewer bank products (3.28 compared with 3.91).
- SHAP analysis highlighted Total_Trans_Ct, Total_Trans_Amt, Total_Revolving_Bal, and Average_Transaction_Amount among the influential features.
- The tuned Gradient Boosting model achieved 96.84% test accuracy, 84.31% recall, 89.54% F1-score, and 99.21% ROC-AUC.
- SHAP explains the model's predictions; it does not prove that a feature causes customer churn.
Retention Cost Analysis
The notebook uses the following hypothetical assumptions, not actual bank figures:
- Retention campaign cost per customer: ₹500
- Estimated value of retaining one customer: ₹5,000
- Expected campaign success rate: 30%
For the 2,026 customers in the test set, the model predicted 287 potential churn customers. Based on the assumptions, the notebook estimated:
Measure	Estimate
Test dataset size	2,026 customers
Predicted churn customers	287
Total retention campaign cost	₹1,43,500
Expected retained customers	Approximately 86
Potential customer value	₹4,30,500
Estimated net financial benefit	₹2,87,000


These are illustrative estimates based on assumed values and are not actual revenue, profit, or guaranteed savings.
8. Challenges Faced
- Hidden missing values: The initial null-value check did not identify missing entries because some categorical fields used the text Unknown. These were handled as missing values during preprocessing.
- Class imbalance: Existing customers substantially outnumbered attrited customers. A balanced Logistic Regression model was compared, and recall and F1-score were included in evaluation.
- Outliers: IQR analysis and boxplots identified extreme observations. They were retained because they may represent genuine customer behaviour and removing them could discard useful information.
- Redundant and unsuitable columns: CLIENTNUM was an identifier, Avg_Open_To_Buy repeated information from other columns, and the two Naive Bayes output columns were excluded from model inputs.
- Model selection: Several classification models were compared, followed by stratified cross-validation and Gradient Boosting hyperparameter tuning.
- Model interpretation: SHAP was used to help explain the final model's predictions.
9. How to Run Locally
Project structure
Arrange your files as follows:
Bank-Churn-Prediction/
├── Bank_Churn_Prediction(6).ipynb
├── dataset/
│   └── BankChurners.csv
├── images/
│   └── project-demo.png      
└── README.md
Setup
Python 3.10 was used in the notebook environment. Create and activate a virtual environment from the project folder.

Windows (Command Prompt):
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate
Install the libraries:
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter shap joblib
Start Jupyter Notebook:
jupyter notebook
Open Bank_Churn_Prediction(6).ipynb and run the cells in order. Ensure that the dataset is saved at dataset/BankChurners.csv before running the data-loading cell. The notebook also contains a model-saving step that saves the trained model as churn_model.pkl when that cell is executed.
10. Limitations and Future Improvements
Limitations
- The project uses one dataset and may not represent every bank or customer population.
- The target classes are imbalanced, and the best prediction threshold may depend on the business cost of missed churners and false alarms.
- Retention campaign costs, customer value, and campaign success rate are hypothetical assumptions.
- SHAP explains model behaviour but does not establish cause and effect.
- The project is a notebook-based analysis; it does not currently include a deployed web application or live prediction service.
Future Improvements
- Validate the model using newer data and data from other customer populations.
- Select a classification threshold based on business costs and retention objectives.
- Explore further model tuning and probability calibration.
- Build an interactive prediction application using Streamlit or Flask.
- Add a repeatable training and prediction pipeline, model monitoring, and data-drift checks.
- Replace the illustrative retention assumptions with verified business data.
11. Tech Stack and Contact Details
Tech stack
- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- SHAP
- Joblib
This project was developed for learning and portfolio demonstration purposes
