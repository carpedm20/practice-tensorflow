import re
import requests
from bs4 import BeautifulSoup as bs4
from nltk.tokenize import RegexpTokenizer

BASE_URL = "http://www.imsdb.com/scripts/%s.html"

movies = [
    "Star Wars: A New Hope", # 1997
    "Star Wars: The Empire Strikes Back", # 1980
    "Star Wars: Return of the Jedi", # 1983
    "Star Wars: The Phantom Menace", # 1999
    "Star Wars: Attack of the Clones", # 2002
    "Star Wars: Revenge of the Sith", # 2005
]

punctuation_remover = RegexpTokenizer(r'\w+')

def clean_text(text):
    return " ".join(text.strip().lower().split())

for movie in movies:
    url = BASE_URL % "-".join(punctuation_remover.tokenize(movie))
    print "="*20
    print url
    r = requests.get(url)

    soup = bs4(r.text.encode('utf-8'))
    text_td = soup.find('td',{'class':'scrtext'})

    # remove useless table
    [t.extract() for t in text_td('table')]

    scripts = []
    for b in text_td.find_all('b'):
        if bool(re.match('^ {37}[^ ].*\n$', b.text)) or \
            bool(re.match('^\t{4}[^\t].*\n$', b.text)):
            character = clean_text(b.text)
            lines = clean_text(b.next_sibling.split('\n\n',1)[0])
            scripts.append((character, lines))

    for script in scripts[:10]:
        print script[0], ":", script[1]
