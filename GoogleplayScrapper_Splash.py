
import json
import pandas as pd
from tqdm import tqdm
import webbrowser
import seaborn as sns
import matplotlib.pyplot as plt
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
from google_play_scraper import Sort, reviews, app
import pycountry
import googletrans
from googletrans import Translator

pycountry.countries.search_fuzzy('Hun')


# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

#-------------------------------------------------------#
#---------------------- Search App in Dubai -------------#
#-------------------------------------------------------#

url = 'https://play.google.com/store/apps'
#webbrowser.open(url, new=2, autoraise=True)
webbrowser.open_new(url)

app_packages = [
    'com.landmarkgroup.splashfashions'
]

app_infos = []

for ap in tqdm(app_packages):
  info = app(ap, lang='en', country='ae')
  del info['comments']
  app_infos.append(info)
  

def print_json(json_object):
  json_str = json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str
  )
  
#print(highlight(json_str, JsonLexer(), TerminalFormatter()))

#print_json(app_infos[0])

app_reviews = []

for ap in tqdm(app_packages):
  for score in list(range(1, 6)):
      for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
        rvs, _ = reviews(
          ap,
          lang='en',
          country='ae',
          sort=sort_order,
          #count= 200 if score == 3 else 100,
          #filter_score_with=score
        )
        for r in rvs:
          r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
          r['appId'] = ap
          r['content']
          #print(r)
        app_reviews.extend(rvs)
      
app_reviews_df = pd.DataFrame(app_reviews)
app_reviews_df = app_reviews_df.drop_duplicates(subset=['content'], keep='first')
app_reviews_df.to_csv('landmarkgroupsplashfashion.csv', index=None, header=True)


#os.chdir('/Users/subhajitchatterjee/Documents/BACKUP/Old Dell Laptop Backup/F Drive/Partnerships/Neostats/Further Ventures')
# from keyphrase_vectorizers import KeyphraseCountVectorizer

# df = pd.read_csv('reviews.csv')