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
#uploaded_file = 'https://drive.google.com/file/d/11xMJmzbOj6Nt48OU1dvhvsdT1C4rTDH6'
data = pd.read_csv('data/Financial_inclusion_dataset.csv')
# Check for outliers
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
# Print all columns
st.write("Columns in data:", financial_data.columns.tolist())
# Remove irrelevant columns
financial_data.columns = financial_data.columns.str.strip().str.lower()
financial_data.drop(columns = ['year', 'uniqueid'], inplace = True, error = 'ignore')
# Select categorical columns to encode
cat_cols = ['country', 'location_type', 'cellphone_access',
            'gender_of_respondent', 'relationship_with_head', 'marital_status',
            'education_level', 'job_type']
# Display the dataset on the deploy page
st.write('Financial Inclusion Dataset', financial_data.sample(100))
# Set the feature for the inputs
def user_inputs():
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
    # Prepare a dataframe for users to input features
    data_features = {'Country': [country],
                          'Location': [location],
                          'Cellphone': [cellphone],
                          'Household_Size': [household_size],
                          'Respondent_Age': [respondent_age],
                          'Gender': [gender],
                          'Relationship': [relationship]}
    user_input = pd.DataFrame(data_features, index = [0])
    return user_input
inputs_df = user_inputs()

# Encode categorical columns
financial_encode = pd.get_dummies(financial_data, columns = cat_cols)
# Balance the data set
from sklearn.utils import resample
# Separate the minority and majority
minority = financial_encode[financial_encode['bank_account'] == 'Yes']
majority = financial_encode[financial_encode['bank_account'] == 'No']
minority_upsampled = resample(minority,
                              replace=True,       # Sample with replacement
                              n_samples=len(majority),  # Match majority class count
                              random_state=42)   # For reproducibility
# Combine majority class with upsampled minority class
balanced_df = pd.concat([majority, minority_upsampled])

# Shuffle the new balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
# Split the data into features(X) and target(y)
X = balanced_df.drop(['bank_account'], axis =1)
y = balanced_df['bank_account']
# Split the data into train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
@ st.cache_resource
def financial_train(X, y):
    # Instantiate StandardScaler
    scaler = StandardScaler()
    # Fit the X_train set into StandardScaler to scale it
    X_train_scaled = scaler.fit_transform(X_train)
    # Instantiate RandomForestClassifier
    clf = RandomForestClassifier()
    # Fit the data into randomforest
    clf.fit(X_train_scaled, y_train)
    return clf
data_train = financial_train(X_train, y_train)
# Create the prediction widget on the app
st.subheader('Predictions')
st.write(inputs_df)
# Encode user_input to match training data
user_input_encoded = pd.get_dummies(inputs_df, columns = inputs_df.select_dtypes(include = 'object').columns)
user_input_encoded = user_input_encoded.reindex(columns = X.columns, fill_value = 0)
# Predict user_input
user_prediction = data_train.predict(user_input_encoded)[0]
st.subheader('User Prediction')
st.success(f'The probability that the customer is likely to churn is {user_prediction}')
