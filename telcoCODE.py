import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df= pd.read_csv('telco.csv')
df.drop('customerID', axis='columns', inplace= True)
print(df.sample(5))
print(df.dtypes)
pd.to_numeric(df.TotalCharges, errors= 'coerce')
df1=df[df.TotalCharges!= ' ']
df1.TotalCharges=pd.to_numeric(df1.TotalCharges)
print(df1.shape, df1.dtypes)
print(df1[df1.Churn=='No'])
tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure
plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
#plt.show()
mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges      
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges      
plt.xlabel("Monthly Charges")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")
plt.hist([mc_churn_yes, mc_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=Yes','Churn=No'])
plt.legend()
#plt.show()
def uniq_val(df):
    for column in df:
        if df[column].dtypes== 'object':
            print(f'{column}: {df[column].unique()}')
df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)
df1.replace({'Female': 1, 'Male': 0}, inplace= True)
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col].replace({'Yes': 1,'No': 0},inplace=True)
for col in df1:
    print(f'{col}: {df1[col].unique()}') 
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])
#df2.to_csv('cleaneddata.csv')
print(df2.dtypes)
#standardazing data:
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_scaled= scaler.fit_transform(df2.drop('Churn', axis= 1))
#modeling, Ill start with linear regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X_scaled, df2['Churn'], test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, y_pred)
print(f'R2 score is={r2}' )
#Since Linear regression didnt showed the best predicts we will try out with other models, 1st will be logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
#lets run cross validation check for it as well:
from sklearn.model_selection import cross_val_score
rf_cv_scores= cross_val_score(model, X_scaled, df2['Churn'],cv= 5, scoring= 'accuracy')
print(f'Logistical regression cross-validation accuracy scores:{rf_cv_scores}')
coeff= model.coef_[0]
feature_names= df2.drop('Churn', axis=1).columns
coeff_df= pd.DataFrame({'Feature': feature_names, 'Coefficient': coeff})
coeff_df['AbsCoefficient']= np.abs(coeff_df['Coefficient'])
coeff_df= coeff_df.sort_values(by='AbsCoefficient', ascending= False)
plt.figure(figsize=(10, 8))
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
#plt.show()
#and for the comparsion with logistic regression i want to use XGBOOST model, since its more advanced and can lead to better accuracy
import xgboost as xgb
xgb_model= xgb.XGBClassifier(n_estimators= 100, max_depth= 3, learning_rate= 0.1, random_state= 42)
xgb_model.fit(X_train, y_train)
y_pred_xgb= xgb_model.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)
print(f'XGBoost Accuracy: {accuracy_xgb}')
print('Confusion Matrix:')
print(conf_matrix_xgb)
print('Classification Report:')
print(class_report_xgb)
cv_scores_xgb = cross_val_score(xgb_model, X_scaled, df2['Churn'], cv=5, scoring='accuracy')
print(f'XGBoost Cross-Validation Accuracy Scores: {cv_scores_xgb}')
print(f'Average CV Accuracy: {cv_scores_xgb.mean()}')
xgb.plot_importance(xgb_model, importance_type='gain', max_num_features=10)  # max_num_features to limit number of features
plt.title('Feature Importance by Gain')
#plt.show()
feature_names = df2.drop('Churn', axis=1).columns

# Plot feature importance with real feature names
importance = xgb_model.get_booster().get_score(importance_type='gain')

# Map feature indices like 'f0', 'f1', ... to actual feature names
importance_mapped = {feature_names[int(k[1:])]: v for k, v in importance.items()}

# Sort by importance values and convert to DataFrame for plotting
importance_df = pd.DataFrame({
    'Feature': list(importance_mapped.keys()),
    'Importance': list(importance_mapped.values())
}).sort_values(by='Importance', ascending=False)

# Plot the feature importance with actual names
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance Score')
plt.title('Feature Importance (XGBoost by Gain)')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()