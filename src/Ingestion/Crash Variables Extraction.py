import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

df = pd.read_csv('month_name1ton.csv')

df['content_clean'] = df['content_clean'].str.lower()

def extract_info(title):
    killed_dead = ''
    injured = ''
    
    keywords = ['killed', 'dead', 'die', 'dies', 'injured']
    allowed_words = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
                     'one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve',
                     'thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty']
    
    for keyword in keywords:
        if keyword in title:
            words_before_keyword = title.split(keyword)[0].strip().split()[-5:]
            filtered_words = [word for word in words_before_keyword if word.lower() in allowed_words]
            filtered_words = ' '.join(filtered_words)
            if filtered_words:  
                if keyword in ['killed', 'dead', 'die', 'dies']:
                    killed_dead = filtered_words
                elif keyword == 'injured':
                    injured = filtered_words
            break
    
    return killed_dead, injured

df['killed/dead'], df['injured'] = zip(*df['Title'].apply(extract_info))

days_to_detect = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
df['day'] = df['content_clean'].str.extract('(' + '|'.join(days_to_detect) + ')', flags=re.IGNORECASE, expand=False)

vehicles_to_detect = [' scooter ',' car ',' tractor ',' truck ',' bus ',' lorry ',' tree ',' van ',' jcb ',' auto ',' bike ',
                      ' autorickshaw ',' suv ',' pickup ',' gorge ',' divider ',' motorcycle ',' loader ',' unidentified ',
                      ' unknown vehicle ',' water tanker ','tanker ',' moped ',' tempo ',' tipper ',' jeep ',' ditch ',
                      ' heavy vehicle ',' dumper ',' ambulance ',' scooty ',' e-rickshaw ',' cycle ',' canal ',' cab ',
                      ' two-wheeler ']

df['vehicle1'] = df['content_clean'].str.extract('(' + '|'.join(vehicles_to_detect) + ')', flags=re.IGNORECASE, expand=False)

df['content_clean'] = df['content_clean'].str.replace('(' + '|'.join(vehicles_to_detect) + ')', '', flags=re.IGNORECASE)

df['vehicle2'] = df['content_clean'].str.findall('(' + '|'.join(vehicles_to_detect) + ')', flags=re.IGNORECASE).apply(lambda x: x[1] if len(x) > 1 else None)

prepositions = ['on', 'at', 'nearby', 'near', 'around']

for i in range(1, 6):
    df[f'location{i}'] = ''

def extract_locations(text):
    locations = {'on': [], 'at': [], 'nearby': [], 'near': [], 'around': []}
    for preposition in prepositions:
        pattern = re.compile(rf'\b{preposition}\s+(?:\w+\s+){{0,5}}\w+', flags=re.IGNORECASE)
        matches = re.findall(pattern, text)
        for match in matches:
            location_words = ', '.join([word.strip("',[]") for word in match.split() if word.strip("',[]") not in stop_words])
            locations[preposition].append(location_words)
    return locations

df['locations'] = df['content_clean'].apply(extract_locations)

for i, preposition in enumerate(prepositions):
    df[f'location{i+1}'] = df['locations'].apply(lambda x: ', '.join([loc.strip("',[]") for loc in x[preposition][:5]]) if preposition in x else '')

df.replace('', 'nil', inplace=True)

df = df.drop('locations', axis=1)

print(df)

df.to_csv('month_namefinal.csv', index=False)
