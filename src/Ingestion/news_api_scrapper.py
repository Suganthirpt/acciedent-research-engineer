from newsapi import NewsApiClient
import pandas as pd
import time

# ==============================
# CONFIGURATION
# ==============================
# 1. PASTE YOUR KEY HERE
API_KEY = '1e09bf4909a14d638bd6244064a04258' 

# 2. Target these Indian News Sites (Comma separated, no spaces)
INDIAN_DOMAINS = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com,deccanherald.com,news18.com,indiatoday.in'

# 3. Search Query (Trucks/Lorries + Crash/Accident)
QUERY = '(truck OR lorry) AND (accident OR crash OR collision) AND (killed OR injured OR dead)'

# 4. Target
TARGET_RECORDS = 50

def get_indian_truck_data():
    print(f"--- Searching for Truck Accidents in India ---")
    try:
        newsapi = NewsApiClient(api_key=API_KEY)
    except Exception as e:
        print("Error: Check your API Key.")
        return

    all_data = []
    page = 1
    
    while len(all_data) < TARGET_RECORDS:
        print(f"Fetching Page {page}...", end=" ")
        
        try:
            response = newsapi.get_everything(
                q=QUERY,
                domains=INDIAN_DOMAINS,  # <--- THIS FORCES INDIAN SOURCES
                language='en',
                sort_by='publishedAt',
                page=page,
                page_size=100
            )
            
            articles = response.get('articles', [])
            
            if not articles:
                print("No more articles found.")
                break
            
            # Filter & Clean
            count = 0
            for art in articles:
                title = art.get('title', '')
                desc = art.get('description', '')
                
                # Strict check: Title must mention the vehicle
                if title and ('truck' in title.lower() or 'lorry' in title.lower() or 'dumper' in title.lower() or 'tanker' in title.lower()):
                    all_data.append({
                        'Date': art.get('publishedAt', '')[:10],
                        'Title': title,
                        'Source': art['source']['name'],
                        'URL': art['url'],
                        'Description': desc
                    })
                    count += 1
            
            print(f"Found {count} valid records. (Total: {len(all_data)})")
            
            if count == 0 and page > 2:
                print("Stopping: Relevance dropping.")
                break
                
            page += 1
            
            # Stop if we hit the target
            if len(all_data) >= TARGET_RECORDS:
                break
                
        except Exception as e:
            print(f"\nAPI Error on Page {page}: {e}")
            break

    # ==============================
    # SAVE
    # ==============================
    if all_data:
        # Trim to 50 if we got too many
        final_df = pd.DataFrame(all_data[:TARGET_RECORDS])
        filename = "India_Truck_Accidents_Final.csv"
        final_df.to_csv(filename, index=False)
        print(f"\nSUCCESS! Saved {len(final_df)} Indian accident records to {filename}")
    else:
        print("\nNo records found. Check your internet connection.")

if __name__ == "__main__":
    get_indian_truck_data()