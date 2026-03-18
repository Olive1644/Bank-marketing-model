#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, roc_curve, accuracy_score, log_loss
#import dataset
bm = pd.read_csv("bank-marketing.csv", sep=";")
print(bm.head(10))
bm.shape
#EDA
print(bm.isnull().sum())
print(bm.duplicated().sum())
#remove disguised missing values
bm.replace('unknown', np.nan, inplace=True)
bm.replace('nonexistent', np.nan, inplace=True)
bm.replace('?', np.nan, inplace=True)
#visualize contact distribution
bm["contact"].value_counts().plot(kind="bar", color=["pink", "cyan", "magenta"])
#visualize month distribution
bm["month"].value_counts().plot(kind="bar", color=["red", "blue", "green", "orange", "purple", "brown", "cyan", "magenta", "yellow", "grey", "black", "pink"])
#visualize day of week distribution
bm["day_of_week"].value_counts().plot(kind="bar", color=["red", "blue", "green", "orange", "purple", "brown", "cyan"])
#visualize loan distribution
bm["loan"].value_counts().plot(kind="bar", color=["red", "blue", "green", "orange"])
#drop redundant features
bm.drop(["poutcome", "default", "previous", "pdays"], axis=1, inplace=True)
#check for missing values
bm.isna().sum()
#check for missing values again after replacing disguised missing values
bm.isnull().sum()
bm.dropna(inplace=True)
#final check of the dataset
bm.info()
#visualize target variable distribution
bm['y'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Subscription Distribution")
plt.xlabel("Subscribed (y)")
plt.ylabel("Count")
plt.show()

#visualize jobs of customers who subscribed to the term deposit
bm[bm['y'] == 'yes']['job'].value_counts().plot(kind='bar', color='orange')
plt.title("Job Distribution of Customers Who Subscribed")
#correlation matrix
plt.figure(figsize=(10,6))
sns.heatmap(bm.corr(numeric_only=True), annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
bm.drop(["cons.price.idx"], axis=1)
#display first few rows of the cleaned dataset
bm.head()
#check number of column and rows
bm.dropna(inplace=True)
bm.describe()
#define x and y
y = bm["y"]
x = bm.drop("y", axis=1)

#get dummies
x = pd.get_dummies(x, drop_first=True)

#map y to binary
y = y.map({'yes':1, 'no':0})
#train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
#build pipelines for xgboost
pipeline_xgb = Pipeline([
    ("scaler", StandardScaler()), 
    ("xgb", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))])
#define parameter grid for xgboost
param_grid_xgb = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.6, 0.8, 1],
    'xgb__colsample_bytree': [0.6, 0.8, 1]
}
#perform randomized search for xgboost
randomxgb = RandomizedSearchCV(pipeline_xgb, param_distributions=param_grid_xgb, cv=5, scoring='neg_log_loss', random_state=42, n_jobs=-1)
#fit the model
randomxgb.fit(x_train, y_train)
#predict and evaluate xgboost
y_pred_xgb = randomxgb.predict(x_test)
y_prob_xgb = randomxgb.predict_proba(x_test)[:, 1]
#evaluate xgboost
print("classification_report:", classification_report(y_test, y_pred_xgb))
print("accuracy_score: ", accuracy_score(y_test, y_pred_xgb))
print("f1_score: ", f1_score(y_test, y_pred_xgb))
print("Log Loss for XGBoost:", log_loss(y_test, y_prob_xgb))
#evaluate xgboost model
y_pred_xgb = randomxgb.predict(x_test)
y_prob_xgb = randomxgb.predict_proba(x_test)[:, 1]
xgb = pd.DataFrame({'Prediction': y_pred_xgb[:38246], 'Probability': y_prob_xgb[:38246], 'Actual': y_test[:38246].values})
print(xgb)
#sort the dataframe by probability in descending order
top_50xgb = xgb.sort_values(by='Probability', ascending=False).head(50)
print(top_50xgb)
#based on the xgb model, the top 2 from
#use RandomForestClassifier
pipeline_rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier())])
param_grid_rf = {
    'rf__bootstrap': [True, False],
    'rf__n_estimators': [50, 100, 200],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}
randomized_rf = RandomizedSearchCV(pipeline_rf, param_distributions = param_grid_rf, cv=5, scoring= 'neg_log_loss', random_state=42)
randomized_rf.fit(x_train, y_train)
y_pred= randomized_rf.predict(x_test)
y_prob= randomized_rf.predict_proba(x_test)[:, 1]
rf = pd.DataFrame({'Prediction': y_pred[:38246], 'Probability': y_prob[:38246], 'Actual': y_test[:38246].values})
print(rf)
#sort values by probability
rf.sort_values(by='Probability', ascending=False)
print("Best Hyperparameters:", randomized_rf.best_params_),
print("classification_report:\n", classification_report(y_test, y_pred)),
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)),
print("f1_score:", f1_score(y_test, y_pred))
print("log_loss:", log_loss(y_test, y_prob))

