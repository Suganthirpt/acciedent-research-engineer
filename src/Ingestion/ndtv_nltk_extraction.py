import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import requests
from bs4 import BeautifulSoup
import os
from tqdm import tqdm

# Enable tqdm for pandas
tqdm.pandas()

# ----------------------------
# 1. Setup & Imports
# ----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Define scraping function
def get_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        
        # NDTV/TOI article body is usually in <p> tags
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# ----------------------------
# 2. Load Data (Corrected for WSL)
# ----------------------------
file_path = "/home/suganthi/hackathon/AccidentParseBS4/TimesOfIndia_Accidents.csv"

if not os.path.exists(file_path):
    print(f"File not found at: {file_path}")
    exit()

df = pd.read_csv(file_path)
print(f"Loaded file with columns: {df.columns.tolist()}")

# ----------------------------
# 3. Fetch Content with tqdm
# ----------------------------
print("⚡ Fetching article text from 'Link' column...")
if "Link" in df.columns:
    # Use progress_apply to show tqdm progress bar
    df["content_clean"] = df["Link"].progress_apply(get_article_text)
    df["content_clean"] = df["content_clean"].fillna("").str.lower()
    print("Extraction complete ✅")
else:
    print("Error: 'Link' column not found in CSV!")


# ----------------------------
# 4. Extract Casualties (From 'Heading')
# ----------------------------
def extract_info(title):
    killed_dead = 'nil'
    injured = 'nil'

    if pd.isna(title):
        return killed_dead, injured

    title = str(title).lower()
    
    # Remove commas to avoid issues like "32," not matching "32"
    title = title.replace(',', '')
    
    keywords = ['killed', 'dead', 'died', 'dies', 'injured']
    words = title.split()

    for keyword in keywords:
        if keyword in words:
            idx = words.index(keyword)
            if idx > 0:
                prev_word = words[idx-1]
                # Check for number digits (1, 2, 10) or words (one, two)
                if prev_word.isdigit() or prev_word in ['one','two','three','four','five']: 
                    # CRITICAL FIX: Avoid ages (e.g., "Woman 24 Dies")
                    # If number is > 10 and < 100, it might be age, but for now we keep it simple.
                    if keyword == 'injured':
                        injured = prev_word
                    else:
                        killed_dead = prev_word
    return killed_dead, injured

if "Heading" in df.columns:
    df[['killed', 'injured']] = df['Heading'].apply(lambda x: pd.Series(extract_info(x)))
else:
    print("Warning: 'Heading' column missing. Skipping casualty extraction.")

# ----------------------------
# 5. Extract Details
# ----------------------------

# ✅ NEW: Get Day directly from 'Time' column (e.g., "Friday February 13")
if "Time" in df.columns:
    # Split by space and take the first word ("Friday")
    df['day'] = df['Time'].astype(str).apply(lambda x: x.split(',')[0].split(' ')[0] if pd.notnull(x) else 'nil')
else:
    df['day'] = 'nil'

# Vehicle Detection (Same as before)
vehicles_list = [
    'scooter','car','tractor','truck','bus','lorry','van','jcb','auto',
    'bike','autorickshaw','suv','pickup','motorcycle','tanker','jeep', 'lamborghini'
]
vehicle_pattern = r'\b(' + '|'.join(map(re.escape, vehicles_list)) + r')\b'

df['vehicle1'] = df['content_clean'].str.extract(vehicle_pattern, expand=False).fillna('nil')

# ----------------------------
# 6. Save File
# ----------------------------
output_path = "/home/suganthi/hackathon/AccidentParseBS4/TimesOfIndia_Accidents_enriched.csv"
df.to_csv(output_path, index=False)

print(f"Saved successfully to: {output_path}")