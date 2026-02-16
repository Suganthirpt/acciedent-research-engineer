# 🚚 Truck Accident Severity Prediction & Risk Scoring
### (NLP Pipeline + ETL + Random Forest Model + Power BI Dashboard)

This project builds a unified accident research pipeline that extracts, processes, and analyzes truck-related accident data using **Kaggle datasets, Mendeley datasets, and news-based NLP extraction**, and produces:

- A cleaned & merged accident dataset  
- A severity prediction model (Minor vs Severe)  
- A probability-based **Risk Score**  
- A fully interactive **Power BI dashboard** to create confusion matrix base for future accident reconstruciton models

---

## 📌 1. Project Objective

The purpose of this project is to build a reusable accident analysis system that can:

1. Consolidate multi‑source accident datasets  
2. Extract crash-related variables using NLP  
3. Train a machine‑learning model to predict accident severity (Minor/Severe)  
4. Generate a **risk score** for reconstructing truck crashes  
5. Visualize insights using an interactive Power BI dashboard  

---

## 📂 2. Folder structure

RAW
├── MoRTH (PDF)
├── OGD (Excel / CSV)
├── Kaggle (CSV)
├── News
|__ news_api
|__ TOI_data
└── Youtube
|__ NDTV data

    src/ingestion
    ├── AccidentParseBS4.ipynb
    ├── Crash Variables Extraction.py
    ├── Data Extraction.py
    ├── indiatoday_extarction_gold_layer.py
    ├── ndtv_extarction_gold.py
    |── ndtv_nltk_extraction.py
    ├── news_api_nltk_extraction.py
    ├── news_api_scrapper.py
    ├── news_nltk_extraction.py
    ├── normalize_news_new.py
    ├── TOI_extraction_gold.py
    |── youtube_scraper.py
    ├── TOI_nlp_extraction.py

Inermediate
|__Kaggle_Cleaned_Data(Normalizing columns)
|__Social-media_cleanded_data(Normalizing columns)

    src/Merging
    |__ Merge_final.py
    |__ ndtv_toi_merge.py
Final
|__ Master_accidents.csv(Unified & Truck only data)

notebooks
|__01_data_inventory.ipynb
|__02. feature_engineering.ipynb
|__03_model_training.ipynb

requirements.txt
|__txt file to capture dependencies


🟧 2. Ingestion Layer (/src/ingestion)
The Ingestion Layer collects raw accident data from structured and unstructured sources and converts them into consistent, analyzable tables. It acts as the foundation of the pipeline and supports both batch and semi‑structured data.
Key Tasks

Scraping accident news from NDTV, TOI, IndiaToday
Extracting incident variables from unstructured text
Pulling accident articles via NewsAPI
Scraping YouTube video metadata
Parsing government datasets (CSV, Excel, PDF)
Standardizing and saving all outputs to /data/intermediate/

Components

AccidentParseBS4.ipynb – HTML parsing of accident stories
Crash Variables Extraction.py – Rule‑based extraction of crash variables
Data Extraction.py – Structured data ingestion engine
indiatoday_extraction_gold_layer.py – IndiaToday scraping
ndtv_extraction_gold.py – NDTV scraping
ndtv_nltk_extraction.py – NLTK extraction for NDTV
news_api_scrapper.py – NewsAPI ingestion
news_api_nltk_extraction.py – NLP processing of NewsAPI text
news_nltk_extraction.py – NLP extraction for general news
normalize_news_new.py – Text cleaning & normalization
TOI_extraction_gold.py – TOI news scraping
TOI_nlp_extraction.py – NLP extraction for TOI articles
youtube_scrapper.py – YouTube metadata ingestion

All ingestion outputs move to the INTERMEDIATE layer for cleaning and merging.

Saves parsed content to a structured table.

## 🧠 3. Data Pipeline Overview

### **Step 1 — Data Collection**
Sources:
- **Kaggle accident datasets** 
- **Mendeley accident dataset (2022 & 2023)** - From Mendeley TOI data 2022 & 2023
- **News articles (NDTV, TOI, NewsAPI)** - Use NLP extraction technique for 2021+ data
- **YouTube video descriptions (scraped)** - last 1 month data


notebooks:

01_data_inventory.ipynb
        The data inventory step evaluates all available data sources, identifies usable truck‑related records, and documents which datasets are included or excluded from the pipeline. This ensures full transparency and reproducibility.

        🔍 1. MoRTH Accident Data (PDF Format)
        Status: ❌ Not used
        Reason:

        Tables are embedded inside PDF and extremely unstructured
        Multiple formatting inconsistencies
        Very low extraction accuracy even after testing:

        pdfplumber
        OCR (Tesseract)
        Not suitable for automated parsing at scale

        🔍 2. Open Government Data (OGD) – Excel/CSV
        Files examined:

        RA2021_A15.csv
        RA2021_A16.csv
        RA2021_A17.csv
        Road_Accidents_2017-Tables_4.1.csv

        Status: ❌ Not used
        Reason:

        Only aggregated statistics available (state‑wise/cause‑wise totals)
        No accident‑level (row‑level) information
        Cannot be used for ML modelling or NLP enrichment


        🔍 3. Kaggle Datasets
        a) accident_prediction_india.csv

        Total records: 3,000
        Truck-related records: 449
        Status: ✔ Used

        b) Road.csv

        Total records: 12,316
        Truck-related records: 2,727
        Status: ✔ Used

        c) AccidentsBig_UK_data.csv

        UK accident data
        Geography not aligned with Indian crash features
        Status: ❌ Excluded

        🔍 4. Mendeley Dataset (2022–2023)
        File: accident_data_22_23.xlsx

        Total records: 2,898
        Truck-related samples (vehicle_1 + vehicle_2): 1,128
        Status: ✔ Used

        🔍 5. YouTube Accident Reports (Scraper Output)

        Accident-related video descriptions scraped
        Total truck-related records: 3
        Helps validate narrative patterns
        Status: ✔ Included in social media dataset

        🔍 6. NewsAPI Accident Articles

        Raw retrieved articles: several
        Truck-related unique articles: 1
        Duplicate of YouTube news
        Status: ❌ Ignored (duplicate content)

        🔍 7. Newspaper Accident Scraping
        Sources:

        NDTV
        Times of India (TOI)
        IndiaToday

        Combined total after processing: ~1,297 accident news records
        Status: ✔ Used in social media dataset
        Used primarily for:

        NLP variable extraction
        Crash descriptions
        Contributing factors

📁 /data/intermediate/social_media_reported_accidents.csv

## **Step 2 — Data Merging**

INTERMEDIATE LAYER
├── Kaggle_Accidents.csv
└── social_media_reported_accidents.csv
            ↓
MERGING LAYER (src/merging)
   ├── Merge_final.py   ---> master_accidents.csv (final dataset)
   └── ndtv_toi_merge.py ---> combined news dataset
            ↓
FINAL LAYER (data/final)
   └── master_accidents.csv

📁 Files in src/merging/



### 1. Merge_final.py
    Purpose:
    This script performs the final merge of all curated datasets into a master file.
    Inputs:

        /data/intermediate/Kaggle_Accidents.csv
        (Structured accident data from Kaggle)
        /data/intermediate/social_media_reported_accidents.csv
        (NLP-processed news + YouTube accident reports)

    Processing steps:

        Reads both datasets into memory
        Standardizes schema (column names & types)
        Aligns accident variables from structured and unstructured sources
        Applies deduplication rules
        Combines all truck-related accident records into one master table

    Output:
        master_accidents.csv

    2. ndtv_toi_merge.py

    Purpose:

    Combines NDTV + TOI + IndiaToday scraped news records
    Creates a consolidated media-based accident dataset
    Ensures consistent formatting before NLP extraction

    which was used in news_paper_dataset before getting the intermediate folder data


### **Step 2 — NLP + Variable Extraction**
Scripts under `/src/ingestion/` extract:
- Vehicle details  
- Road condition  
- Weather & lighting  
- Crash variables mentioned in text  

### **Step 3 — Data Merging**
    'Merge_final.py` and notebooks combine NLP outputs + structured datasets → `/data/intermediate/`.

### **Step 4 — Feature Engineering**
    '02_feature_engineering.ipynb`:

    🟦 1. Load Master Dataset
    Input file:
    /data/final/master_accidents.csv

    Initial data schema included:
    'accident_id', 'source', 'year', 'month', 'day_of_week', 'state',
    'vehicle_1', 'vehicle_2', 'vehicles_involved', 'party_age_group',
    'victim_age_group', 'cause_of_accident', 'weather_condition',
    'road_condition', 'lighting_condition', 'fatalities', 'injuries',
    'casualties', 'severity'

    🟩 2. Schema Description

    Data types reviewed (categorical, numeric)
    Unique values distribution checked
    Summary statistics generated for numerical fields

    🟧 3. Null Value Assessment

    Null counts computed per column
    Missingness patterns studied
    Categorical vs numeric missingness handled separately

    🟥 4. Dropping Unnecessary Columns
    The following fields were removed due to low predictive value or redundancy:

    year — highly imbalanced, not useful for severity modeling
    casualties — fully derived from fatalities + injuries

    🟦 5. State Name Normalization

    Standardized state spellings (e.g., Tamil Nadu vs T.N.)
    Handled abbreviations and case inconsistencies
    Ensured consistent geographic categorization

    🟪 6. Month Imputation (Season-Based Strategy)
    Instead of dropping records with missing month:

    Mapped seasons in India:

    Summer
    Monsoon
    Post‑monsoon
    Winter


    Used weather_condition + lighting_condition patterns to infer month
    Ensured robust imputation for records from news text where month was missing

🟫 7. Fatalities & Injuries Outlier Review

    Outliers analyzed using boxplots
    No upper outliers removed (severe accidents may have extreme counts)
    Lower bounds thoroughly checked
    Confirmed: no negative values present → dataset valid
        
🟩 8. Vehicle_1 & Vehicle_2 Normalization
    To support truck‑specific modeling:

    All heavy vehicles mapped to “Truck”
    Light vehicles grouped under generic categories (Car, Bike, Bus, Auto, etc.)
    Improved classification consistency across:

    Kaggle data
    Mendeley data
    Newspaper data


🟨 9. Vehicles Involved

    Standardized count of vehicles involved
    Useful for predicting severity (multi‑vehicle crashes often more severe)


🟫 10. Party Age Group (Driver Age Group)

    Social media/news datasets did not contain this attribute
    Standardized missing values to "Unknown"

🟫 11. Victim Age Group

    Similar strategy as driver age group
    Missing values standardized to "Unknown"

🟥 12. Cause of Accident Normalization
Originally ~40 different cause labels from multiple sources:

Consolidated into 9 major standardized groups:
    Examples:

    Over‑speeding
    Drunk driving
    Mechanical failure
    Weather issue
    Road defect
    Poor visibility
    Collision with animal
    Tailgating
    Unspecified/Other

To improve machine learning performance.

🟦 13. Weather Condition → Weather Group
    Mapped all weather variants into 4 main categories:

    Clear
    Rainy
    Foggy
    Other

    Used this grouping for:

    Month imputation
    Severity modeling


🟨 14. Road Condition Standardization

    Unified road surface terms (dry, wet, slushy, potholes)
    Imputed missing values based on common patterns

🟩 15. Lighting Condition Standardization

    Mapped conditions into:

    Daylight
    Night (with street lights)
    Night (dark)
    Dawn/Dusk


    Imputed missing values using time‑of‑day clues from news data

🟥 16. Severity Column Normalization
    Severity labels consolidated into three standard classes:

    Minor
    Severe
    Fatal

    Prepared these classes for:

    Binary conversion (Minor vs Severe)
    Risk Score modeling

🟦 17. Export Transformed Dataset

    /data/final/accident_ml_data.csv.csv

🧾 Final Data Schema (Post Feature Engineering)
    After completing all cleaning, normalization, imputations, and standardization steps in 02_feature_engineering.ipynb, the final ML‑ready dataset contains the following fields: 
    'accident_id',
    'source',
    'year',
    'month',
    'day_of_week',
    'state',
    'vehicle_1',
    'vehicle_2',
    'vehicles_involved',
    'party_age_group',
    'victim_age_group',
    'cause_of_accident',
    'road_condition',
    'lighting_condition',
    'fatalities',
    'injuries',
    'casualties',
    'severity',
    'weather_group'

    descriptions for each data type will be added to report.docx


### **Step 5 — Model Training**
`03_model_training.ipynb` trains a **Random Forest classifier** that outputs:
- `Predicted_Severity` (Minor/Severe)  
- `Risk_Score` (probability of Severe)

Model saved as `temp_model.json`.

### **Step 6 — Dashboard Preparation**
`Accident_Risk_Dashboard_Data.csv` feeds the **Power BI dashboard**.


---
🤖 Model Training (03_model_training.ipynb)
Note: All visualizations (confusion matrix, SHAP plots, factor charts) are included in report.docx

Input file:
/data/final/accident_ml_data.csv

Columns removed (not used in ML features):
casualties (derived from fatalities + injuries)
year (low signal)
accident_id (identifier)
source (non-predictive)

Post‑drop feature set (15 columns):
['month', 'day_of_week', 'state', 'vehicle_1', 'vehicle_2', 
 'vehicles_involved', 'party_age_group', 'victim_age_group',
 'cause_of_accident', 'road_condition', 'lighting_condition',
 'fatalities', 'injuries', 'severity', 'weather_group']

2) Feature Typing

Categorical (cat_cols):
'month', 'day_of_week', 'state', 'vehicle_1', 'vehicle_2',
'party_age_group', 'victim_age_group', 'cause_of_accident',
'weather_group', 'road_condition', 'lighting_condition', 'severity'

Numeric (num_cols):
'vehicles_involved', 'fatalities', 'injuries'

3) Encoding Strategy

Custom (non‑ordinal) mappings applied for:

month, day_of_week, severity, party_age_group, victim_age_group
(Prevents incorrect ordinal relationships)

Label Encoding used for other categoricals.
Encoded categories:

state: 33 classes
vehicle_1: 15 classes
vehicle_2: 16 classes
cause_of_accident: 9 classes
weather_group: 4 classes
road_condition: 3 classes
lighting_condition: 3 classes


✅ All columns converted to numeric successfully.

Category maps (examples):

state → 33 normalized states (e.g., Andhra Pradesh … West Bengal)
vehicle_1 / vehicle_2 → standardized types (Truck, Car, Bus, Two Wheeler, etc.)
cause_of_accident → 9 groups:

Drunk/Drug Driving, Lane Violation, Loss of Control, No Distancing/Rear End, Parking Issue, Reckless Driving, Severe Collision, Speeding, Unknown/Other

weather_group → Clear, Cold, Fog/Mist, Rainy
road_condition → Dry, Wet, Damaged/Construction
lighting_condition → Daylight, Twilight, Dark

4) Train/Test Split

Split: 80/20
Train: 3,578 rows
Test: 895 rows
Features used in final training: 12 (after dropping non-feature fields)

casualties (derived from fatalities + injuries)
year (low signal)
accident_id (identifier)
source (non-predictive)

5) Severity Target Setup
The original severity had 3 classes (Minor, Serious, Fatal).
Due to class imbalance and poor recall for Serious, we created a binary target:

severity_binary:

0 = Minor
1 = Severe (Serious + Fatal)

Class counts:

Before (3‑class)

Fatal = 2,511
Serious = 1,423
Minor = 539

After (binary)

Minor (0) = 2,511
Severe (1) = 1,962


6) Models Trained & Benchmarks
We trained three models and compared performance on the binary setup (0=Minor, 1=Severe):

Logistic Regression

Accuracy: 83.02%
Minor: P=0.80, R=0.95, F1=0.86
Severe: P=0.90, R=0.67, F1=0.77

Random Forest (class_weight='balanced')

Accuracy: ~82.57% (best RF run)
Minor: P=0.82, R=0.89, F1=0.85
Severe: P=0.83, R=0.74, F1=0.78

XGBoost

Accuracy: 84.47%
Minor: P=0.82, R=0.94, F1=0.87
Severe: P=0.89, R=0.72, F1=0.80
(Grid search used: best params {learning_rate: 0.01, max_depth: 3, n_estimators: 300} for cause model; severity used standard tuned settings.)

7) Risk Score Generation

Algorithm: Random Forest Classifier
Output:

Predicted_Severity (0/1 → mapped to Minor/Severe)
Risk_Score = Probability of Severe (class 1) scaled to 0–100 for dashboard readability

Examples from test set:

High Risk (> 65): Several rows with Risk_Score in 65–73 range
Low Risk (< 20): Many rows 0–18 range

✅ Exported 895 test rows (no leakage) for Power BI confusion matrix:

/data/final/Full_Accident_Risk_Analysis.csv


8) Interpretability

Logistic Regression coefficients (top contributors):

↑ Severity: cause_of_accident, lighting_condition, weather_group
↓ Severity: victim_age_group, party_age_group, vehicle_1, state, vehicle_2


SHAP (XGBoost)

SHAP values computed on test data to explain key drivers per record (runtime ~ few seconds)

9) Auxiliary Model — Cause of Accident Prediction (9 classes)

Random Forest & XGBoost baselines

Top‑1 Accuracy: ~40–41%
Top‑3 Accuracy: ~88% (actionable suggestion list)


Reason: Multiple causes share overlapping patterns (e.g., Speeding vs Reckless Driving).
Decision: Not used in dashboard scoring, but can assist analysts via top‑3 suggestions.

**Metrics:**  
- Confusion Matrix  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Threshold tuning using What‑If parameter in Power BI

## 📊 5. Power BI Dashboard

The dashboard includes:

### **1. Risk Score Threshold Slider**
Tune classification threshold (0.1–0.9).

### **2. Confusion Matrix**
Shows TP/FP/TN/FN dynamically.

### **3. Risk Score Distribution**
Shows severity probability spread.

File: /data/final/Full_Accident_Risk_Analysis.csv (test set only)
Use cases in dashboard:

Threshold‑tunable Confusion Matrix (via What‑If parameter)
Risk Score distribution
Feature vs Avg Risk (weather/road/lighting/day_of_week)
Map by state (Avg risk or severe %)

### **Install dependencies:**
```bash
pip install -r requirements.txt