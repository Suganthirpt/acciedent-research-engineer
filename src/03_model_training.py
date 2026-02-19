# %%
import pandas as pd
import numpy as np
import os

df_final = pd.read_csv("data/final/accident_ml_data.csv")
print(df_final.shape)
print(df_final.columns)

# %%
df_final.drop(columns = ['Unnamed: 0','casualties','year','accident_id','source'],inplace = True, errors ='ignore')
print("Columns dropped. Remaining columns:", df_final.columns.tolist())
print(df_final.shape)
df_final.head()

# %%
df_final['month'].value_counts()

# %% [markdown]
# # Building ML Model 

# %% [markdown]
# ## 1. Categorize columns 

# %% [markdown]
# ### Step 1: Separate Categorical vs. Numerical

# %%
from sklearn.preprocessing import LabelEncoder

# 1. Define columns
cat_cols = [
    'month', 'day_of_week', 'state', 
    'vehicle_1', 'vehicle_2', 
    'party_age_group', 'victim_age_group', 
    'cause_of_accident', 
    'weather_group', 'road_condition', 'lighting_condition',
    'severity'
]

# 2. Ensure Numerical columns are numbers
num_cols = ['vehicles_involved', 'fatalities', 'injuries']
for col in num_cols:
    df_final[col] = pd.to_numeric(df_final[col], errors='coerce').fillna(0)

print("Numerical columns fixed.")

# %% [markdown]
# ### Step 2: Apply String Indexing (Label Encoding)

# %%
# Define custom mappings
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}

day_map = {
    'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
    'Friday': 5, 'Saturday': 6
}

severity_map = {
    'Minor': 0, 
    'Serious': 1, 
    'Fatal': 2
}

# Fix age groups to be sequential rather than alphabetical
age_map = {
    'Under 18': 0,
    '18-30': 1,
    '31-50': 2,
    'Over 51': 3,
    'Unknown': -1  # Or handle as a separate category/mode
}

# applying custom mappings for temporal data
df_final['month'] = df_final['month'].map(month_map)
df_final['day_of_week'] = df_final['day_of_week'].map(day_map)
df_final['severity'] = df_final['severity'].map(severity_map)
df_final['party_age_group'] = df_final['party_age_group'].map(age_map)
df_final['victim_age_group'] = df_final['victim_age_group'].map(age_map)

# %%
# Create a dictionary to save mappings
label_encoders = {}

cat_cols_2 = [
    'state', 'vehicle_1', 'vehicle_2', 
    'cause_of_accident','weather_group', 
    'road_condition', 'lighting_condition',
]
for col in cat_cols_2:
    # Initialize Encoder
    label = LabelEncoder()
    
    # Fit and Transform (Convert Text to Numbers)
    # .astype(str) if there is a number
    df_final[col] = label.fit_transform(df_final[col].astype(str))
    
    # Save the encoder for later lookup
    label_encoders[col] = label
    
    print(f"Encoded {col}: {len(label.classes_)} unique classes")

print("\nAll columns are now numeric!")

# %%
for col, le in label_encoders.items():
    # Get the mapping: Index (0,1,2) -> Class Name ('High', 'Low', 'Med')
    mapping = dict(zip(range(len(le.classes_)), le.classes_))
    
    print(f"\nFeature: {col}")
    print(mapping)

# %%
df_final.describe()

# %% [markdown]
# ### RANDOM FOREST CLASSIFIER ALGORITHM FOR SEVEARITY MODEL PREDICITON

# %% [markdown]
# #### Step 1: Define X (Features) and y (Target)

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Define X (Features) - Dropping Target + Leakage Columns
# We drop 'fatalities' and 'injuries' to force the model to predict based on CAUSES, not OUTCOMES.
X = df_final.drop(columns=['severity_binary','severity', 'fatalities', 'injuries'], errors='ignore')

# 2. Define y (Target)
y = df_final['severity']

# 3. Split: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_test = X_test.rename(columns={'weather_condition': 'weather_group'})
print(f"Data Split Successfully!")
print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

# %%
X_train.columns
X_test.columns

# %%
# df_final['severity_binary'].value_counts()

# %% [markdown]
# #### Step 2: Train the Model

# %%
# 3. Initialize the Model
# n_estimators=100 means we use 100 decision trees
rf_binary= RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# 4. Train the Model (Fit)
print("Training the model...")
rf_binary.fit(X_train, y_train)
# rf_binary.fit(X_train, y_train)

print("Model Trained!")

# %% [markdown]
# #### Step 3: Evaluate Performance

# %%
# 5. Make Predictions
y_pred = rf_binary.predict(X_test)

# 6. Check Accuracy and Detailed Report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nDetailed Classification Report:")
# We use the encoder to get the actual names (Fatal, Minor, etc.) instead of 0, 1, 2
target_names = ['Minor','Serious','Fatal']
print(classification_report(y_test, y_pred,target_names=target_names))

# %% [markdown]
# #### There are 3 classes the model prediction is not good for servere class. Strategy to build better model
# ##### Class 0 (Low Risk): Minor
# ##### Class 1 (High Risk): Serious + Fatal

# %% [markdown]
# #### Code to Merge and Retrain the sevearity columns

# %%
# Current: {0: 'Minor', 1: 'Serious', 2: 'Fatal'}

# New Logic:
# 1 (Fatal)   -> 1 (Severe)
# 2 (Serious) -> 1 (Severe)
# 0 (Minor)   -> 0 (Minor)

# Define the mapping dictionary
severity_remap = {0: 0, 2: 1, 1: 1}

# Create a new column 'severity_binary'
df_final['severity_binary'] = df_final['severity'].map(severity_remap)

# Verify the new counts
print("Old Counts (0=Fatal, 1=Minor, 2=Serious):")
print(df_final['severity'].value_counts())

print("\nNew Counts (0=Minor, 1=Severe):")
print(df_final['severity_binary'].value_counts())

# %% [markdown]
# ### Retraining the Random Forest Classifier model with new severity column

# %%
# 1. Define x (Target)
X = df_final.drop(columns=['severity_binary','severity', 'fatalities', 'injuries'], errors='ignore')

# 1. Define y (Target)
y = df_final['severity_binary']

# 3. Split: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data Split Successfully!")
print(f"Training Data: {X_train.shape}")
print(f"Testing Data: {X_test.shape}")

rf_binary = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 10, 1: 1})

# 4. Train the Model (Fit)
print("Training the model...")
rf_binary.fit(X_train, y_train)

print("Model Trained!")

# 5. Make Predictions
y_pred = rf_binary.predict(X_test)

# 6. Check Accuracy and Detailed Report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.2%}")

print("\nDetailed Classification Report:")
# We use the encoder to get the actual names (Fatal, Minor, etc.) instead of 0, 1, 2
target_names = ['Minor','Severe']
print(classification_report(y_test, y_pred,target_names=target_names))
X_train.columns

# %% [markdown]
# #### Generate Risk Scores for Your Data

# %%
# 1. Get Probability of 'Severe' (Class 1)

# X_test = X_test.rename(columns={'weather_group': 'weather_condition'})
risk_probabilities = rf_binary.predict_proba(X_test)[:, 1]


# Predict on test data
y_pred_binary = rf_binary.predict(X_test)

# 2. Convert to Score (0-100)
risk_scores = risk_probabilities * 100

# 3. Create a DataFrame to View Results
risk_df = pd.DataFrame({
    'Actual_Severity': y_test,
    'Predicted_Severity': y_pred_binary,
    'Risk_Score': risk_scores.round(1) # Round to 1 decimal place
})

# Show High Risk Examples
print("---High Risk Examples (Risk > 65) ---")
print(risk_df[risk_df['Risk_Score'] > 65].head(10))

# Show Low Risk Examples
print("\n--- ✅ Low Risk Examples (Risk < 20) ---")
print(risk_df[risk_df['Risk_Score'] < 20].head(10))


# # %%
# import pandas as pd
# import matplotlib.pyplot as plt

# # Get feature importances
# importances = rf_binary.feature_importances_
# feature_names = X_train.columns

# # Create a DataFrame
# feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)

# # Plot
# plt.figure(figsize=(10, 6))
# plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'])
# plt.gca().invert_yaxis()
# plt.title("What is driving the predictions?")
# plt.show()

# %% [markdown]
# ### Observation: Model accuracy and prediction improved after changing it to binary class. 
# #### For Binary class variable logistic regression is better than Random Forest Classifier.

# %% [markdown]
# #### LOGISTIC REGRESSION MODEL FOR SEVEARITY PREDICITON

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 1. Scale the Data (Crucial for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train the Model
# class_weight='balanced' helps it catch the 'Severe' cases
log_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
log_model.fit(X_train_scaled, y_train)

# 3. Evaluate
y_pred_log = log_model.predict(X_test_scaled)

print(f"📊 Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=['Minor', 'Severe']))

# 4. The "Magic" Part: Interpreting the Coefficients
coeffs = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
})

# Sort by impact (High Positive = Causes Severity, High Negative = Prevents it)
coeffs = coeffs.sort_values(by='Coefficient', ascending=False)

print("\nTop 5 Factors Increasing Severity:")
print(coeffs.head(5))

print("\nTop 5 Factors Decreasing Severity (Predicting Minor):")
print(coeffs.tail(5))

# %% [markdown]
# #### Step 1: Severity Prediction with XGBoost

# %%
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Initialize the XGBoost Classifier
# use_label_encoder=False removes a warning in newer versions
# eval_metric='logloss' is standard for binary classification
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    random_state=42, 
    use_label_encoder=False, 
    eval_metric='logloss'
)

# 2. Train the Model
print(" Training XGBoost Model...")
xgb_model.fit(X_train, y_train)

# 3. Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# 4. Evaluate Performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"\n XGBoost Accuracy: {accuracy_xgb:.2%}")

print("\n--- Classification Report (0=Minor, 1=Severe) ---")
target_names = ['Minor', 'Severe']
print(classification_report(y_test, y_pred_xgb, target_names=target_names))

# 5. Compare with Random Forest (Optional check)
# If your RF was ~95%, let's see if XGBoost beats it!

# # %%
# # Plot Feature Importance
# plt.figure(figsize=(10, 8))
# # max_num_features=10 shows only the top 10
# plot_importance(xgb_model, max_num_features=10, height=0.5, color='teal')
# plt.title('XGBoost Feature Importance (Top 10 Drivers of Severity)')
# plt.show()

# # %%
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. Day of Week Impact
# plt.figure(figsize=(10, 5))
# # Assuming 'day_of_week' is textual (Mon, Tue...) or numeric
# sns.countplot(x='day_of_week', hue='severity_binary', data=df_final, palette='viridis')
# plt.title('Accident Severity by Day of Week')
# plt.show()


# %% [markdown]
# ### OBSERVATION:FRIDAYS WERE MOST OFTHE ACCIDENTS HAPPENED DUE TO INTER CITY MOVEMENT AHEAD OF WEEKEND

# %% [markdown]
# # Building a "Cause of Accident" Model

# %% [markdown]
# ## Step 1: Random Forest Approach

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Define Features (X) and New Target (y)
# Drop 'cause_of_accident' (target) and outcome columns (severity, fatalities, etc.)
X_cause = df_final.drop(columns=['cause_of_accident', 'severity', 'severity_binary', 'fatalities', 'injuries', 'casualties'], errors='ignore')
y_cause = df_final['cause_of_accident']

# 2. Split Data
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cause, y_cause, test_size=0.2, random_state=42)

# 3. Train Model
# n_estimators=100 is standard
# class_weight='balanced' is CRITICAL because causes like "Drunk Driving" might be rarer than "Speeding"
rf_cause = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("Training Cause Prediction Model...")
rf_cause.fit(X_train_c, y_train_c)

# 4. Evaluate
y_pred_cause = rf_cause.predict(X_test_c)
print(f"Cause Prediction Accuracy: {accuracy_score(y_test_c, y_pred_cause):.2%}")

# 5. Show Detailed Report (To see which causes are easy/hard to predict)
# We need the class names from the specific encoder for 'cause_of_accident'
# If you lost the encoder, use: target_names=y_cause.unique().astype(str) (sorted)
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_cause))

# %% [markdown]
# Due to 9 classes the accuracy went down and there is data imbalance for class 0, 1,4.

# %% [markdown]
# Predicting the exact cause is extremely difficult because causes like 'Speeding' and 'Reckless Driving' have identical patterns. However, our model still correctly identifies the top 3 most likely causes with 76% accuracy (Top-3 Accuracy)."

# %% [markdown]
# # Step 4: Explainable AI (SHAP)

# %%
import shap
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 1. Prepare Data (Dropping 'state' and leaky features)
X = df_final.drop(columns=['severity_binary', 'severity', 'fatalities', 'injuries', 'state'], errors='ignore')
y = df_final['severity_binary']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Model
model_xgb = XGBClassifier(
    n_estimators=100, 
    random_state=42, 
    scale_pos_weight=(len(y[y==0])/len(y[y==1])) # to handle imbalance
)
model_xgb.fit(X_train, y_train)
print("Model retrained successfully!")

# --- SHAP SECTION ---

# 4. Get the exact feature names the model was trained on
expected_features = X_train.columns.tolist()

# 5. Define prediction function for KernelExplainer
def predict_fn(data):
    
    if not isinstance(data, pd.DataFrame): # If SHAP passes a numpy array, convert back to DataFrame with correct names
        data = pd.DataFrame(data, columns=expected_features)
    return model_xgb.predict_proba(data)

# 6. Initialize Explainer using the actual X_train

explainer = shap.KernelExplainer(predict_fn, X_train.iloc[:20]) # Using 20 rows of background data keeps the KernelExplainer efficient

print("Calculating SHAP values (this may take a minute)...")

shap_values = explainer.shap_values(X_test.iloc[:50]) # Calculate for first 50 test rows

print("SHAP Values calculated successfully!")

# # 7. Summary Plot
# # shap_values[1] represents the impact on the 'Severe' class
# shap.summary_plot(shap_values[1], X_test.iloc[:50])

# # %%
# import matplotlib.pyplot as plt

# # Check which class is which (e.g., Class 0 = Drunk, Class 1 = Speeding)
# # We will plot the SHAP values for Class 1 (adjust index if needed)
# class_index = 1 

# plt.title(f'Top Factors specific to Cause Class {class_index}', fontsize=14)
# shap.summary_plot(shap_values[class_index], X_test_c.iloc[:50], plot_type="bar")

# %% [markdown]
# ## Export data to visualize in Power BI dashboard

# %%
# --- Manual Inverse Maps ---
inv_month = {v: k for k, v in month_map.items()}
inv_day = {v: k for k, v in day_map.items()}
# Mapping for your 0/1 binary model
inv_severity_bin = {0: 'Minor', 1: 'Severe'} 
# Mapping for age groups
inv_party_age = {v: k for k, v in age_map.items()}
inv_victim_age = {v: k for k, v in age_map.items()}

# --- Label Encoder Inverse Maps (Based on your provided lists) ---
inv_state = {0: 'Andhra Pradesh', 1: 'Arunachal Pradesh', 2: 'Assam', 3: 'Bihar', 4: 'Chandigarh', 5: 'Chhattisgarh', 6: 'Delhi', 7: 'Goa', 8: 'Gujarat', 9: 'Haryana', 10: 'Himachal Pradesh', 11: 'Jammu and Kashmir', 12: 'Jharkhand', 13: 'Karnataka', 14: 'Kerala', 15: 'Madhya Pradesh', 16: 'Maharashtra', 17: 'Manipur', 18: 'Meghalaya', 19: 'Mizoram', 20: 'Nagaland', 21: 'Odisha', 22: 'Puducherry', 23: 'Punjab', 24: 'Rajasthan', 25: 'Sikkim', 26: 'Tamil Nadu', 27: 'Telangana', 28: 'Tripura', 29: 'Unknown', 30: 'Uttar Pradesh', 31: 'Uttarakhand', 32: 'West Bengal'}

inv_vehicle_1 = {0: 'Ambulance', 1: 'Auto Rickshaw', 2: 'Bicycle', 3: 'Bus', 4: 'Car', 5: 'Crane', 6: 'Jugaad Vehicle', 7: 'Pedestrian', 8: 'Pick Up', 9: 'Suv', 10: 'Train', 11: 'Truck', 12: 'Two Wheeler', 13: 'Van', 14: 'Volvo'}

inv_vehicle_2 = {0: 'Ambulance', 1: 'Auto', 2: 'Auto Rickshaw', 3: 'Bicycle', 4: 'Bus', 5: 'Canal', 6: 'Car', 7: 'Ditch', 8: 'Divider', 9: 'Gorge', 10: 'River', 11: 'Truck', 12: 'Two_Wheeler', 13: 'Unidentified', 14: 'Van', 15: 'Wall'}

inv_cause = {0: 'Drunk/Drug Driving', 1: 'Lane Violation', 2: 'Loss of Control', 3: 'No Distancing/Rear End', 4: 'Parking Issue', 5: 'Reckless Driving', 6: 'Severe Collision', 7: 'Speeding', 8: 'Unknown/Other'}

inv_weather = {0: 'Clear', 1: 'Cold', 2: 'Fog/Mist', 3: 'Rainy'}

inv_road = {0: 'Damaged/Construction', 1: 'Dry', 2: 'Wet'}

inv_lighting = {0: 'Dark', 1: 'Daylight', 2: 'Twilight'}

# %%
import pandas as pd
import numpy as np

# 1. Create a Master DataFrame for ALL rows
# We use the full index of X_test to ensure no data is left behind
dashboard_df = X_test_c.loc[X_test.index].copy()

# 2. Define Inverse Mappings (Manual & Label Enc)
inv_month = {v: k for k, v in month_map.items()}
inv_day = {v: k for k, v in day_map.items()}
inv_severity_bin = {0: 'Minor', 1: 'Severe'} 
inv_age = {v: k for k, v in age_map.items()}

inv_state = {0: 'Andhra Pradesh', 1: 'Arunachal Pradesh', 2: 'Assam', 3: 'Bihar', 4: 'Chandigarh', 5: 'Chhattisgarh', 6: 'Delhi', 7: 'Goa', 8: 'Gujarat', 9: 'Haryana', 10: 'Himachal Pradesh', 11: 'Jammu and Kashmir', 12: 'Jharkhand', 13: 'Karnataka', 14: 'Kerala', 15: 'Madhya Pradesh', 16: 'Maharashtra', 17: 'Manipur', 18: 'Meghalaya', 19: 'Mizoram', 20: 'Nagaland', 21: 'Odisha', 22: 'Puducherry', 23: 'Punjab', 24: 'Rajasthan', 25: 'Sikkim', 26: 'Tamil Nadu', 27: 'Telangana', 28: 'Tripura', 29: 'Unknown', 30: 'Uttar Pradesh', 31: 'Uttarakhand', 32: 'West Bengal'}
inv_vehicle_1 = {0: 'Ambulance', 1: 'Auto Rickshaw', 2: 'Bicycle', 3: 'Bus', 4: 'Car', 5: 'Crane', 6: 'Jugaad Vehicle', 7: 'Pedestrian', 8: 'Pick Up', 9: 'Suv', 10: 'Train', 11: 'Truck', 12: 'Two Wheeler', 13: 'Van', 14: 'Volvo'}
inv_vehicle_2 = {0: 'Ambulance', 1: 'Auto', 2: 'Auto Rickshaw', 3: 'Bicycle', 4: 'Bus', 5: 'Canal', 6: 'Car', 7: 'Ditch', 8: 'Divider', 9: 'Gorge', 10: 'River', 11: 'Truck', 12: 'Two_Wheeler', 13: 'Unidentified', 14: 'Van', 15: 'Wall'}
inv_cause = {0: 'Drunk/Drug Driving', 1: 'Lane Violation', 2: 'Loss of Control', 3: 'No Distancing/Rear End', 4: 'Parking Issue', 5: 'Reckless Driving', 6: 'Severe Collision', 7: 'Speeding', 8: 'Unknown/Other'}
inv_weather = {0: 'Clear', 1: 'Cold', 2: 'Fog/Mist', 3: 'Rainy'}
inv_road = {0: 'Damaged/Construction', 1: 'Dry', 2: 'Wet'}
inv_lighting = {0: 'Dark', 1: 'Daylight', 2: 'Twilight'}

# 3. Apply Inverse Transformations to the full dataframe
transform_logic = {
    'month': inv_month,
    'day_of_week': inv_day,
    'state': inv_state,
    'vehicle_1': inv_vehicle_1,
    'vehicle_2': inv_vehicle_2,
    'party_age_group': inv_age,
    'victim_age_group': inv_age,
    'cause_of_accident': inv_cause,
    'weather_group': inv_weather,
    'road_condition': inv_road,
    'lighting_condition': inv_lighting,
}

for col, mapping in transform_logic.items():
    if col in dashboard_df.columns:
        dashboard_df[col] = dashboard_df[col].map(mapping)

# 4. Add Predictions & Risk Scores (Corrected Syntax)
dashboard_df['Actual_Severity'] = y_test.map(inv_severity_bin).values
# Creating a Series for y_pred ensures it aligns with the dataframe index
dashboard_df['Predicted_Severity'] = pd.Series(y_pred, index=y_test.index).map(inv_severity_bin).values
dashboard_df['Risk_Score'] = risk_scores

# 5. Save the Full Dataset
dashboard_df.to_csv("output/Full_Accident_Risk_Analysis.csv", index=False)
print(f"✅ Success! {len(dashboard_df)} rows exported for Power BI.")


