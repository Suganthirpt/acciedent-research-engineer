# FINAL MERGING OF 2 DATASETS AND DEFINING DATA SCHEMA
# ==============================
# 1. CONFIGURATION
# ==============================

import pandas as pd
import os
import numpy as np

# Get the directory where Merge_final.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

FILE_KAGGLE = os.path.join(current_dir, "../../data/intermediate/Kaggle_Accidents.csv")
FILE_SCRAPED = os.path.join(current_dir, "../../data/intermediate/social_media_reported_accidents.csv")

OUTPUT_FILE = os.path.join(current_dir, "../../data/final/master_accidents.csv")


# THE FINAL MASTER SCHEMA
GOLDEN_COLUMNS = [
    'accident_id', 'source', 
    'year', 'month', 'day_of_week', 
    'state', 
    'vehicle_1', 'vehicle_2', 'vehicles_involved',
    'party_age_group',     
    'victim_age_group',  
    'cause_of_accident',     
    'weather_condition', 'road_condition', 'lighting_condition',
    'fatalities', 'injuries', 'casualties', 'severity'
]

# ==============================
# 2. LOAD DATA
# ==============================
try:
    df_kaggle = pd.read_csv(FILE_KAGGLE)
    df_scraped = pd.read_csv(FILE_SCRAPED)
    print(df_kaggle.shape)
    print(df_scraped.shape)

except FileNotFoundError:
    print("Error: Intermediate files not found. Check paths.")
    exit()

# ==============================
# 3. TRANSFORM KAGGLE DATA
# ==============================
print("Transforming Kaggle Data...")

# 1. Rename vehicle_type -> vehicle_1
df_kaggle = df_kaggle.rename(columns={'vehicle_type': 'vehicle_1'})

# 2. Add Missing Columns
df_kaggle['vehicle_2'] = 'Unknown'
# Note: Kaggle already has 'injuries' calculated? If not, calculate it:
if 'injuries' not in df_kaggle.columns:
    # Calculate difference
    df_kaggle['injuries'] = df_kaggle['casualties'] - df_kaggle['fatalities']
    
    # Force negative numbers to 0
    df_kaggle['injuries'] = df_kaggle['injuries'].clip(lower=0)

# 3. Drop Unwanted Columns
cols_to_drop = ['time', 'city_area', 'url', 'date'] # Drop if they exist
df_kaggle = df_kaggle.drop(columns=[c for c in cols_to_drop if c in df_kaggle.columns])

# ==============================
# 4. TRANSFORM SCRAPED DATA
# ==============================
print("Transforming Scraped Data...")

# 1. Calculate Severity (The Logic: Fatal > 0 -> Fatal, Injuries > 2 -> Serious, Else -> Minor)
def calc_severity(row):
    if row['fatalities'] > 0:
        return 'Fatal'
    elif row['injuries'] > 2:
        return 'Serious'
    else:
        return 'Minor'

df_scraped['severity'] = df_scraped.apply(calc_severity, axis=1)

# 2. Fill Missing Columns (Unknowns)
df_scraped['vehicle_2'] = df_scraped['vehicle_2'].fillna('Unknown')
df_scraped['party_age_group'] = 'Unknown'
df_scraped['road_condition'] = 'Unknown'
df_scraped['lighting_condition'] = 'Unknown'
df_scraped['vehicles_involved'] = 'Unknown' # or np.nan if you prefer numeric

# 3. Ensure Year/Month exist (Extract from Date before dropping it)
# (Assuming your intermediate file already has year/month, but just in case:)
if 'year' not in df_scraped.columns and 'date' in df_scraped.columns:
    df_scraped['date'] = pd.to_datetime(df_scraped['date'])
    df_scraped['year'] = df_scraped['date'].dt.year
    df_scraped['month'] = df_scraped['date'].dt.month_name()

# 4. Drop Unwanted Columns
cols_to_drop_scraped = ['date', 'time', 'city_area', 'url']
df_scraped = df_scraped.drop(columns=[c for c in cols_to_drop_scraped if c in df_scraped.columns])

# ==============================
# 5. MERGE & FINALIZE
# ==============================
print("Merging into Master Dataset...")

master_df = pd.concat([df_kaggle, df_scraped], ignore_index=True)

# 1. Ensure Golden Schema
master_df = master_df[GOLDEN_COLUMNS]

# 2. Final Cleanup (Fill NaNs with Unknown for categorical, 0 for numeric)
cat_cols = ['state', 'vehicle_1', 'vehicle_2', 'cause_of_accident', 
            'weather_condition', 'road_condition', 'lighting_condition']
num_cols = ['fatalities', 'injuries', 'casualties', 'year']

for col in cat_cols:
    master_df[col] = master_df[col].fillna('Unknown').astype(str).str.title().str.strip()
    master_df[col] = master_df[col].replace({'Nan': 'Unknown', '': 'Unknown'})

for col in num_cols:
    master_df[col] = master_df[col].fillna(0).astype(int)

# 3. Save
master_df.to_csv(OUTPUT_FILE, index=False)

print(f"SUCCESS! Master Dataset created at: {OUTPUT_FILE}")
print(f"Total Rows: {len(master_df)}")
print("\nSample Data:")
print(master_df[['source', 'severity', 'vehicle_1', 'vehicle_2', 'injuries']].head(10))
