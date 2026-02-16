import pandas as pd

# ---------------------------------------------------------
# 1. Load the Datasets
# ---------------------------------------------------------
file_indiatoday = "/home/suganthi/hackathon/AccidentParseBS4/IndiaToday/indiatoday_HCV_Accidents_Final.csv"
file_ndtv = "/home/suganthi/hackathon/AccidentParseBS4/NDTV_data/NDTV_HCV_Accidents_Final.csv"

try:
    df_it = pd.read_csv(file_indiatoday)
    df_ndtv = pd.read_csv(file_ndtv)
    print("✅ Files loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    # Create dummy data for testing if files don't exist (remove this in production)
    exit()

# ---------------------------------------------------------
# 2. Combine Data
# ---------------------------------------------------------
# Merge both dataframes
df_all = pd.concat([df_it, df_ndtv], ignore_index=True)

# ---------------------------------------------------------
# 3. Standardize Date & Year
# ---------------------------------------------------------
# Convert Date to datetime objects
df_all['Date'] = pd.to_datetime(df_all['Date'], errors='coerce')

# Ensure 'Year' column is accurate (derived from Date)
df_all['Year'] = df_all['Date'].dt.year

# ---------------------------------------------------------
# 4. FILTER: Remove 2022 and 2023
# ---------------------------------------------------------
# Keep only years that are NOT 2022 and NOT 2023
df_filtered = df_all[~df_all['Year'].isin([2022, 2023])].copy()

print(f"📉 Filtered out {len(df_all) - len(df_filtered)} rows from 2022 & 2023.")

# ---------------------------------------------------------
# 5. Deduplicate (Based on Date & State)
# ---------------------------------------------------------
# Create a temporary clean state column for comparison
df_filtered['State_Clean'] = df_filtered['State'].astype(str).str.lower().str.strip()

# Check for duplicates based on Date and State
# keep='first' retains the first occurrence, marks others as True
duplicate_mask = df_filtered.duplicated(subset=['Date', 'State_Clean'], keep='first')

# Separate the unique and duplicate data
df_final = df_filtered[~duplicate_mask].copy()
df_duplicates = df_filtered[duplicate_mask].copy()

# Drop helper column
df_final.drop(columns=['State_Clean'], inplace=True)

# ---------------------------------------------------------
# 6. Save Outputs
# ---------------------------------------------------------
output_filename = "Final_Unique_HCV_Accidents_Cleaned.csv"
df_final.to_csv(output_filename, index=False)

print("-" * 30)
print(f"Original Total Rows:   {len(df_all)}")
print(f"Rows After Year Filter:{len(df_filtered)}")
print(f"Final Unique Rows:     {len(df_final)}")
print("-" * 30)
print(f"✅ Saved final data to: {output_filename}")