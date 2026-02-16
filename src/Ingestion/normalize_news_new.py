import pandas as pd
import numpy as np
import os

# 1. Get the folder where THIS script lives (src/scrapper)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path: Go up 2 levels, then into data/raw
csv_path = os.path.join(current_dir, "../../data/raw/Final_Unique_HCV_Accidents_Cleaned.csv")
output_path = os.path.join(current_dir, "../../data/raw/HCV_Accidents_normalized.csv")
# 3. Load it
df_new = pd.read_csv(csv_path)

print("Processing New Source Data...")

# 1. Create a copy
df_normalized = df_new.copy()

# 2. Rename columns to match the "Old schema"
rename_map = {
    'Year': 'year',
    'Month': 'month',
    'Date': 'date',
    'Day_of_Week': 'day_of_week',
    'State': 'state',
    'City': 'city_area',
    'Fatalities': 'fatalities',
    'Injuries': 'injuries',
    'Vehicle_1': 'vehicle_1',
    'Vehicle_2': 'vehicle_2',
    'Victim_age_band': 'victim_age_group',
    'Cause_of_accident': 'cause_of_accident',
    'Link': 'url'
}
 # 'party_age' and 'Heading' are dropped or kept as extra depending on your need
df_normalized = df_normalized.rename(columns=rename_map)

# 3. Add columns that exist in Old Schema but are missing in New Data
# 'casualties' is usually Fatalities + Injuries
# Ensure columns are numeric before summing
df_normalized['fatalities'] = pd.to_numeric(df_normalized['fatalities'], errors='coerce').fillna(0)
df_normalized['injuries'] = pd.to_numeric(df_normalized['injuries'], errors='coerce').fillna(0)

df_normalized['casualties'] = df_normalized['fatalities'] + df_normalized['injuries']

# 'weather_condition' is missing in new source, so we fill with None/NaN
df_normalized['weather_condition'] = np.nan

# 4. Enforce the exact column order of the Old Source
target_columns = [
    'accident_id', 'source', 'year', 'month', 'date', 'day_of_week',
    'state', 'city_area', 'fatalities', 'injuries', 'casualties',
    'vehicle_1', 'vehicle_2', 'victim_age_group', 'cause_of_accident',
    'weather_condition', 'url'
]

# This selects only the matching columns and puts them in the correct order
df_final = df_normalized[target_columns]

print("Normalization Complete. Columns are now:")
print(df_final.columns.tolist())

# 6. SAVE


df_final.to_csv(output_path, index=False)


print(f"df_final_SUCCESS! Merged {len(df_final)} records.")
