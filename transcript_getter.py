import os
import requests
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def get_script_url():
    index_url = #website url
    html = requests.get(index_url)
    soup = BeautifulSoup(html.content, 'html.parser')
    links = soup.find('div', {'class': 'content'}).find_all('a')
    episodes = [link for link in links][1:-2]  # Exclude copyright, season 8, and pdf links
    return episodes

def clean_script_text(script_text, episode):
    # Clean up script text by removing title, end, and filler words
    if fuzz.partial_ratio(script_text[0], episode.text) > 60:
        script_text[0] = ''
    if 'end' in script_text[-1].lower():
        script_text[-1] = ''
    cut_words = ['written by', 'directed by', 'transcript by', 'transcribed by']
    for i in range(4):
        for word in cut_words:
            if word.lower() in script_text[i].lower():
                script_text[i] = ''
    cleaned_script = ' '.join([x for x in script_text if x])
    cleaned_script = unicodeToAscii(cleaned_script)
    return cleaned_script

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def main():
    filename = 'gg_script2.txt'
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    if os.path.exists(output_path):
        return  # File already exists

    episodes = get_script_url()

    with open(output_path, 'w', encoding="utf-8") as f:
        for episode in episodes:
            index = episode.get('href')
            html = requests.get(index)
            soup = BeautifulSoup(html.content, 'html.parser')
            script = soup.find('div', {'class': 'content'}).get_text(strip=True, separator='\n').splitlines()
            script = list(filter(None, script))
            script = clean_script_text(script, episode)
            f.write(script + '\n')
    print('file downloaded')

if __name__ == "__main__":
    main()