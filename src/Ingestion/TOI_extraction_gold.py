import pandas as pd
import re
import numpy as np

# Load your file
df = pd.read_csv("TimesOfIndia_Accidents_enriched.csv")

# ---------------------------------------------------------
# 1. Basic Setup & Source
# ---------------------------------------------------------
df['source'] = "Times of India"

# ---------------------------------------------------------
# 2. Date Processing
# ---------------------------------------------------------
# Convert 'Time' column to datetime objects
df['date_obj'] = pd.to_datetime(df['Time'], errors='coerce')

df['Date'] = df['date_obj'].dt.date
df['Year'] = df['date_obj'].dt.year
df['Month'] = df['date_obj'].dt.month_name()
df['Day_of_Week'] = df['date_obj'].dt.day_name()

# ---------------------------------------------------------
# 3. Location (State & City)
# ---------------------------------------------------------
indian_states = {
    'andhra': 'Andhra Pradesh', 'telangana': 'Telangana', 'karnataka': 'Karnataka',
    'tamil nadu': 'Tamil Nadu', 'kerala': 'Kerala', 'maharashtra': 'Maharashtra',
    'delhi': 'Delhi', 'up': 'Uttar Pradesh', 'uttar pradesh': 'Uttar Pradesh',
    'haryana': 'Haryana', 'punjab': 'Punjab', 'rajasthan': 'Rajasthan',
    'gujarat': 'Gujarat', 'mp': 'Madhya Pradesh', 'bihar': 'Bihar',
    'bengal': 'West Bengal', 'odisha': 'Odisha', 'assam': 'Assam',
    'himachal': 'Himachal Pradesh', 'uttarakhand': 'Uttarakhand',
    'chhattisgarh': 'Chhattisgarh', 'jharkhand': 'Jharkhand', 'goa': 'Goa'
}

cities_to_state = {
    'hyderabad': 'Telangana', 'bengaluru': 'Karnataka', 'bangalore': 'Karnataka',
    'chennai': 'Tamil Nadu', 'mumbai': 'Maharashtra', 'pune': 'Maharashtra',
    'delhi': 'Delhi', 'new delhi': 'Delhi', 'noida': 'Uttar Pradesh',
    'gurugram': 'Haryana', 'gurgaon': 'Haryana', 'kolkata': 'West Bengal',
    'lucknow': 'Uttar Pradesh', 'kanpur': 'Uttar Pradesh', 'jaipur': 'Rajasthan',
    'ahmedabad': 'Gujarat', 'surat': 'Gujarat', 'indore': 'Madhya Pradesh',
    'bhopal': 'Madhya Pradesh', 'patna': 'Bihar', 'ludhiana': 'Punjab',
    'agra': 'Uttar Pradesh', 'nashik': 'Maharashtra', 'nagpur': 'Maharashtra',
    'vadodara': 'Gujarat', 'ghaziabad': 'Uttar Pradesh', 'thane': 'Maharashtra',
    'meerut': 'Uttar Pradesh', 'rajkot': 'Gujarat', 'varanasi': 'Uttar Pradesh',
    'srinagar': 'Jammu & Kashmir', 'aurangabad': 'Maharashtra', 'dhanbad': 'Jharkhand',
    'amritsar': 'Punjab', 'navi mumbai': 'Maharashtra', 'allahabad': 'Uttar Pradesh',
    'prayagraj': 'Uttar Pradesh', 'howrah': 'West Bengal', 'jabalpur': 'Madhya Pradesh',
    'gwalior': 'Madhya Pradesh', 'vijayawada': 'Andhra Pradesh', 'jodhpur': 'Rajasthan',
    'coimbatore': 'Tamil Nadu'
}

def extract_location(text):
    if pd.isna(text): return "Unknown", "Unknown"
    text = str(text).lower()
    found_city = "Unknown"
    found_state = "Unknown"

    # Check cities first (more specific)
    for city, state in cities_to_state.items():
        if re.search(r'\b' + re.escape(city) + r'\b', text):
            found_city = city.title()
            found_state = state
            break 

    # If state not found via city, look for state name directly
    if found_state == "Unknown":
        for state_key, state_name in indian_states.items():
            if re.search(r'\b' + re.escape(state_key) + r'\b', text):
                found_state = state_name
                break

    return found_state, found_city

df[['State', 'City']] = df['Heading'].apply(lambda x: pd.Series(extract_location(x)))

# ---------------------------------------------------------
# 4. Vehicles (Vehicle_1 = Heavy, Vehicle_2 = Other)
# ---------------------------------------------------------
hcv_list = [
    'truck', 'container', 'tanker', 'lorry', 'dumper', 'trailer',
    'tipper', 'mixer', 'jcb', 'crane', 'bus', 'volvo', 'loader',
    'heavy vehicle', 'tractor', 'dcm', 'canter', 'mini truck'
]

other_vehicles_list = [
    'car', 'bike', 'scooter', 'motorcycle', 'auto', 'rickshaw',
    'suv', 'jeep', 'van', 'cab', 'taxi', 'ambulance', 'cycle',
    'bicycle', 'pedestrian', 'scooty', 'moped', 'bolero', 'scorpio',
    'fortuner', 'innova', 'swift', 'alto', 'wagonr', 'city', 'verna',
    'creta', 'baleno', 'dzire', 'e-rickshaw', 'lamborghini'
]

def extract_vehicles_split(text):
    if pd.isna(text): return "Unknown", "Unknown"
    text = str(text).lower()
    
    found_hcvs = []
    found_others = []

    # Find HCVs
    for v in hcv_list:
        if re.search(r'\b' + re.escape(v) + r'\b', text):
            found_hcvs.append(v)
            
    # Find Others
    for v in other_vehicles_list:
        if re.search(r'\b' + re.escape(v) + r'\b', text):
            found_others.append(v)
            
    # Logic: Vehicle 1 is priority (HCV), Vehicle 2 is secondary
    v1, v2 = "Unknown", "Unknown"
    
    all_hcv = sorted(list(set(found_hcvs)))
    all_other = sorted(list(set(found_others)))
    
    if all_hcv:
        v1 = all_hcv[0] # First HCV found
        if len(all_hcv) > 1:
            v2 = all_hcv[1] # Second HCV
        elif all_other:
            v2 = all_other[0] # First other vehicle
    elif all_other:
        v1 = all_other[0] # No HCV
        if len(all_other) > 1:
            v2 = all_other[1]
            
    return v1, v2

df[['Vehicle_1', 'Vehicle_2']] = df['Heading'].apply(lambda x: pd.Series(extract_vehicles_split(x)))

# ---------------------------------------------------------
# 5. Casualties
# ---------------------------------------------------------
def extract_casualty(text, keywords):
    if pd.isna(text): return 0
    text = str(text).lower()
    num_map = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5}
    for k in keywords:
        match = re.search(r'(\d+|one|two|three|four|five)\s+(?:people|persons)?\s*' + k, text)
        if match:
            val = match.group(1)
            return int(val) if val.isdigit() else num_map.get(val, 0)
    # Singular check (e.g., "Driver killed")
    singular_keywords = ['man', 'woman', 'boy', 'girl', 'driver', 'rider', 'person', 'cleaner', 'helper']
    if any(k in text for k in keywords):
        for subj in singular_keywords:
             if f"{subj} {k}" in text or f"{subj} dies" in text: return 1
        return 1 # Fallback
    return 0

df['Fatalities'] = df['Heading'].apply(lambda x: extract_casualty(x, ['killed', 'dead', 'dies', 'death', 'crushed']))
df['Injuries'] = df['Heading'].apply(lambda x: extract_casualty(x, ['injured', 'hurt', 'critical']))

# ---------------------------------------------------------
# 6. Age & Age Band
# ---------------------------------------------------------
def extract_age(text):
    if pd.isna(text): return None
    text = str(text).lower()
    # Pattern 1: "Man, 32"
    match = re.search(r'(?:man|woman|boy|girl|driver|victim|techie|student)[,\s]+(\d{1,2})\b', text)
    if match: return int(match.group(1))
    # Pattern 2: "32-year-old"
    match2 = re.search(r'(\d{1,2})[-\s]year[-\s]old', text)
    if match2: return int(match2.group(1))
    return None

df['party_age'] = df['Heading'].apply(extract_age)

def get_age_band(age):
    if pd.isna(age): return "Unknown"
    age = int(age)
    if age <= 18: return "0-18"
    elif age <= 30: return "19-30"
    elif age <= 45: return "31-45"
    elif age <= 60: return "46-60"
    else: return "60+"

df['Victim_age_band'] = df['party_age'].apply(get_age_band)

# ---------------------------------------------------------
# 7. Cause of Accident
# ---------------------------------------------------------
cause_keywords = {
    'Overspeeding': ['speeding', 'overspeeding', 'fast', 'racing', 'rash'],
    'Drunk Driving': ['drunk', 'alcohol', 'liquor'],
    'Fog/Weather': ['fog', 'rain', 'mist', 'visibility'],
    'Brake Failure': ['brake fail'],
    'Wrong Side': ['wrong side', 'wrong way'],
    'Hit and Run': ['hit and run', 'fled'],
    'Overturning': ['overturn', 'topple', 'flipped'],
    'Collision': ['collision', 'collide', 'hit', 'rammed', 'crash', 'crushed']
}

def extract_cause(text):
    if pd.isna(text): return "Unknown"
    text = str(text).lower()
    for cause, keywords in cause_keywords.items():
        for k in keywords:
            if k in text:
                return cause
    return "Unknown"

df['Cause_of_accident'] = df['Heading'].apply(extract_cause)

# ---------------------------------------------------------
# 8. Filter & ID Creation
# ---------------------------------------------------------
# Filter: Keep only rows where Vehicle_1 OR Vehicle_2 is a Heavy Vehicle
mask_v1_hcv = df['Vehicle_1'].isin(hcv_list)
mask_v2_hcv = df['Vehicle_2'].isin(hcv_list)

df_hcv = df[mask_v1_hcv | mask_v2_hcv].copy()

# Create Accident ID
df_hcv['accident_id'] = 'ACC_' + df_hcv['Year'].astype(str) + '_' + df_hcv.index.astype(str).str.zfill(4)

# Select final columns
final_columns = [
    'accident_id', 'source', 'Year', 'Month', 'Date', 'Day_of_Week',
    'State', 'City', 'Fatalities', 'Injuries', 'Vehicle_1', 'Vehicle_2',
    'Victim_age_band', 'party_age', 'Cause_of_accident', 'Heading', 'Link'
]

df_final = df_hcv[final_columns]
output_filename = "TOI_HCV_Accidents_Final.csv"
df_final.to_csv(output_filename, index=False)

print(f"Processed {len(df_final)} heavy vehicle accidents.")
print(f"Saved to: {output_filename}")