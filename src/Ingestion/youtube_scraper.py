from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# ==============================
# CONFIGURATION
# ==============================
API_KEY = 'AIzaSyAiF4587rrzlpMe-eG1V40Z9zKt4lgRucs' 

TARGET_RECORDS = 50
MIN_VIEWS = 10000
SEARCH_QUERY = "truck accident news india"

def get_video_data():
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    collected_data = []
    next_page_token = None
    
    print(f"--- Searching YouTube for: '{SEARCH_QUERY}' ---")

    while len(collected_data) < TARGET_RECORDS:
        
        # 1. Search
        try:
            search_response = youtube.search().list(
                q=SEARCH_QUERY,
                part='id,snippet',
                maxResults=50,
                type='video',
                regionCode='IN',
                order='relevance', # Relevance often finds better news clips than 'date'
                pageToken=next_page_token
            ).execute()
        except Exception as e:
            print(f"Search Error: {e}")
            break

        video_ids = [item['id']['videoId'] for item in search_response['items']]
        
        # 2. Get Stats (Views)
        stats_response = youtube.videos().list(
            part='statistics,contentDetails,snippet', # We get snippet again for full description
            id=','.join(video_ids)
        ).execute()
        
        # Map IDs to details
        vid_details = {item['id']: item for item in stats_response['items']}

        # 3. Process
        for item in search_response['items']:
            vid_id = item['id']['videoId']
            
            # Get details from the second API call
            details = vid_details.get(vid_id)
            if not details: continue
            
            title = details['snippet']['title']
            description = details['snippet']['description'] # <--- GOLD MINE
            views = int(details['statistics'].get('viewCount', 0))
            
            if views < MIN_VIEWS:
                continue

            print(f"Processing: {title[:40]}...", end=" ")

            # --- STRATEGY: TRY TRANSCRIPT, FALLBACK TO DESCRIPTION ---
            full_text = ""
            source_type = ""

            try:
                # Try fetching transcript
                transcript_list = YouTubeTranscriptApi.list_transcripts(vid_id)
                # Prefer English/Hindi, translate to English
                transcript = transcript_list.find_transcript(['en', 'hi', 'ta', 'te', 'kn', 'ml'])
                translated_transcript = transcript.translate('en').fetch()
                full_text = " ".join([t['text'] for t in translated_transcript])
                source_type = "Transcript"
                
                # Combine with description for maximum context
                full_text = description + " " + full_text

            except (TranscriptsDisabled, NoTranscriptFound, Exception):
                # FALLBACK: Use Description only
                full_text = description
                source_type = "Description Only"

            # Clean text
            full_text = full_text.replace("\n", " ")

            # Extract Stats
            killed, injured = extract_stats_from_text(full_text)
            
            # SAVE CONDITION: Save if we found stats OR if keywords exist
            if killed > 0 or injured > 0 or 'truck' in title.lower():
                collected_data.append({
                    'Video_ID': vid_id,
                    'Title': title,
                    'Views': views,
                    'Date': details['snippet']['publishedAt'][:10],
                    'Source_Type': source_type,
                    'Killed': killed,
                    'Injured': injured,
                    'Text_Preview': full_text[:300] + "...",
                    'URL': f"https://www.youtube.com/watch?v={vid_id}"
                })
                print(f"[{source_type}] -> Saved (K:{killed}, I:{injured})")
            else:
                print("[SKIPPED] Irrelevant")

            if len(collected_data) >= TARGET_RECORDS:
                break
        
        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            break

    # ==============================
    # SAVE
    # ==============================
    if collected_data:
        df = pd.DataFrame(collected_data)
        df.to_csv('YouTube_Truck_Accidents_Hybrid.csv', index=False)
        print(f"\nSUCCESS! Saved {len(df)} records to 'YouTube_Truck_Accidents_Hybrid.csv'")
    else:
        print("No records found.")

# --- Extraction Logic ---
def extract_stats_from_text(text):
    text = text.lower()
    killed = 0
    injured = 0
    
    w2n = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}
    
    # KILLED
    k_pattern = re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?(killed|dead|died|death)', re.IGNORECASE)
    match_k = k_pattern.search(text)
    if match_k:
        num = match_k.group(1)
        killed = int(num) if num.isdigit() else w2n.get(num, 0)

    # INJURED
    i_pattern = re.compile(r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+)?(injured|hurt|injuries)', re.IGNORECASE)
    match_i = i_pattern.search(text)
    if match_i:
        num = match_i.group(1)
        injured = int(num) if num.isdigit() else w2n.get(num, 0)
        
    return killed, injured

if __name__ == "__main__":
    get_video_data()