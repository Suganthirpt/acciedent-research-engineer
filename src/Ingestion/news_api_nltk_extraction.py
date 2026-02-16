import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# --- CONFIGURATION ---
INPUT_FILE = 'India_Truck_Accidents_Final.csv'
OUTPUT_FILE = 'India_Truck_Accidents_Processed.csv'

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 1. Load Data
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records from {INPUT_FILE}")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please check the file name.")
    exit()

# 2. Prepare Text for Analysis
# We combine Title + Description to get more details (like location/numbers)
# treating NaN as empty strings
df['content_clean'] = (df['Title'].fillna('') + ' ' + df['Description'].fillna('')).str.lower()


# 3. Extract Killed / Injured
def extract_stats(text):
    killed = 0
    injured = 0
    
    # Mapping words to numbers
    word_to_num = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20
    }
    
    # Helper to convert a word or digit string to an integer
    def parse_number(num_str):
        if num_str.isdigit():
            return int(num_str)
        return word_to_num.get(num_str, 0)

    # Keywords
    death_keywords = ['killed', 'dead', 'die', 'dies', 'death', 'deaths']
    injury_keywords = ['injured', 'injury', 'injuries', 'hurt']

    # Regex logic: Look for "Number + Keyword" (e.g., "2 killed", "two dead")
    # We look for a number (digits or words) followed by up to 2 words, then the keyword
    
    # CHECK DEATHS
    for kw in death_keywords:
        # Pattern: (number) (optional word) (keyword)
        # matches: "2 killed", "two people dead", "3 feared dead"
        pattern = re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?' + kw, re.IGNORECASE)
        match = pattern.search(text)
        if match:
            killed = parse_number(match.group(1))
            break # Found a stat, stop looking

    # CHECK INJURIES
    for kw in injury_keywords:
        pattern = re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?' + kw, re.IGNORECASE)
        match = pattern.search(text)
        if match:
            injured = parse_number(match.group(1))
            break 

    return killed, injured

# Apply extraction
stats = df['content_clean'].apply(extract_stats)
df['Killed'], df['Injured'] = zip(*stats)


# 4. Extract Vehicles (Vehicle 1 & Vehicle 2)
vehicles_list = [
    'scooter', 'car', 'tractor', 'truck', 'bus', 'lorry', 'van', 'jcb', 
    'auto', 'bike', 'autorickshaw', 'suv', 'pickup', 'motorcycle', 
    'tanker', 'moped', 'tempo', 'tipper', 'jeep', 'dumper', 'ambulance', 
    'scooty', 'e-rickshaw', 'cycle', 'cab', 'trailer', 'container'
]

def extract_vehicles(text):
    found = []
    # Using set to avoid duplicates like "truck hit truck" showing up as ['truck']
    # But if it's "truck hit car", we want both.
    
    # Regex to find all vehicles
    pattern = r'\b(' + '|'.join(vehicles_list) + r')\b'
    matches = re.findall(pattern, text)
    
    # Remove duplicates while preserving order
    seen = set()
    for v in matches:
        if v not in seen:
            found.append(v)
            seen.add(v)
            
    v1 = found[0] if len(found) > 0 else 'unknown'
    v2 = found[1] if len(found) > 1 else 'nil'
    return v1, v2

vehs = df['content_clean'].apply(extract_vehicles)
df['Vehicle_1'], df['Vehicle_2'] = zip(*vehs)


# 5. Extract Location (Simple Preposition Check)
def extract_location(text):
    # Look for "at [Location]" or "near [Location]"
    # Captures up to 4 words after the preposition
    pattern = re.compile(r'\b(at|near|near by|on)\s+((?:[A-Z][a-z]+\s*){1,4})')
    
    # We use the original Title/Description (Mixed Case) to help identify Proper Nouns (places)
    # But since we lowercased everything earlier, we fallback to a simpler word count strategy
    
    # Regex: "near" + next 1-4 words excluding stopwords
    pattern_simple = re.compile(r'\b(near|at|in)\s+((?:\w+\s*){1,4})', re.IGNORECASE)
    match = pattern_simple.search(text)
    
    if match:
        loc_raw = match.group(2).strip()
        # Clean up (remove "the", "a")
        loc_clean = ' '.join([w for w in loc_raw.split() if w not in stop_words])
        return loc_clean
    return 'nil'

df['Location'] = df['content_clean'].apply(extract_location)


# 6. Save Final
final_columns = ['Date', 'Title', 'Killed', 'Injured', 'Vehicle_1', 'Vehicle_2', 'Location', 'URL']
df_final = df[final_columns]

print("\n--- SAMPLE EXTRACTED DATA ---")
print(df_final.head())

df_final.to_csv(OUTPUT_FILE, index=False)
print(f"\nSuccess! Processed data saved to: {OUTPUT_FILE}")