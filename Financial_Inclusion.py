#### Import the libraries
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.title('Financial Inclusion in Africa Prediction')
st.write('This model use RandomForestClassifier to make prediction and because the data is highly imbalance, I used RandomUnderSample to modify it')

# Create user input (widgets)
st.sidebar.header("Input features for prediction")

# Load the dataset
#uploaded_file = st.file_uploader('Desktop/GMC/ML/Financial_inclusion_dataset.csv', type = 'csv')
data = pd.read_csv(r'Desktop/GMC/ML/Financial_inclusion_dataset.csv')
# Chedk for outliers
# Z-score method (for normally distributed data)
from scipy import stats
numerical_columns = data.select_dtypes(include = ['int', 'float']).columns
z_scores = stats.zscore(data[numerical_columns])
outliers = (abs(z_scores) > 3)  # Threshold of 3 standard deviations
# IQR method (robust to non-normal distributions)
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = ((data[numerical_columns] < (Q1 - 1.5*IQR)) | ((data[numerical_columns] > (Q3 + 1.5*IQR))))
# Remove outliers
financial_data = data[~outliers_iqr.any(axis=1)]
# Remove irrelevant columns
financial_data.drop(columns = ['year', 'uniqueid'], inplace = True)
# Select categorical columns to encode
cat_cols = ['country', 'location_type', 'cellphone_access',
            'gender_of_respondent', 'relationship_with_head', 'marital_status',
            'education_level', 'job_type']
# Display the dataset on the deploy page
st.write('Financial Inclusion Dataset', financial_data.sample(100))
# Set the feature for the inputs
country = st.sidebar.selectbox('Country', financial_data['country'].unique())
location = st.sidebar.selectbox('Location', financial_data['location_type'].unique())
cellphone = st.sidebar.selectbox('Cellphone', financial_data['cellphone_access'].unique())
household_size = st.sidebar.slider('Household_Size',
                                int(financial_data['household_size'].min()),
                                int(financial_data['household_size'].mean()),
                                int(financial_data['household_size'].max()))
respondent_age = st.sidebar.slider('Respondent_Age',
                                int(financial_data['age_of_respondent'].min()),
                                int(financial_data['age_of_respondent'].mean()),
                                int(financial_data['age_of_respondent'].max()))
gender = st.sidebar.selectbox('Gender', financial_data['gender_of_respondent'].unique())
relationship = st.sidebar.selectbox('Relationship_With_Head', financial_data['relationship_with_head'])
# Encode categorical columns
financial_encode = pd.get_dummies(financial_data, columns = cat_cols)
# Split the data into features(X) and target(y)
X = financial_encode.drop(['bank_account'], axis =1)
y = financial_encode['bank_account']
# Split the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
@ st.cache_resource
def financial_train(X, y):
    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit the X_train set into StandardScaler to scale it
    X_train_scaled = scaler.fit_transform(X_train)
    # Instantiate RandomForestClassifier and RandomUnderSampler
    rus = RandomUnderSampler()
    clf = RandomForestClassifier()
    # Fit the RandomUnderSampler
    X_resampled, y_resampled = rus.fit_resample(X_train_scaled, y_train)
    # Fit the data into randomforest
    clf.fit(X_resampled, y_resampled)
    return clf
data_train = financial_train(X_train, y_train)
# Predict the data
results = X_test.copy()
predict = data_train.predict(X_test)
results['predicted_churn'] = predict
# Create the prediction widget on the app
st.subheader('Predictions')
st.write(results.head())
# Prepare a DataFrame for the user inputs features
user_input = pd.DataFrame({'Country': [country],
                          'Location': [location],
                          'Cellphone': [cellphone],
                          'Household_Size': [household_size],
                          'Respondent_Age': [respondent_age],
                          'Gender': [gender],
                          'Relationship': [relationship]})
# Encode user_input to match training data
user_input_encoded = pd.get_dummies(user_input, columns = user_input.select_dtypes(include = 'object').columns)
user_input_encoded = user_input_encoded.reindex(columns = X.columns, fill_value = 0)
# Predict user_input
user_prediction = data_train.predict(user_input_encoded)[0]
st.subheader('User Prediction')
st.success(f'The probability that the customer is likely to churn is {user_prediction}')