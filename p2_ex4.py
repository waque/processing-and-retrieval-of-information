from table import *
from utils import *
from parser import *
import re
from pageRanker import PageRanker
from utils2 import *
from eval import *
from sklearn.externals import joblib
from p2_ex3 import classifier_summary

rss = ['http://rss.nytimes.com/services/xml/rss/nyt/World.xml','http://feeds.washingtonpost.com/rss/rss_blogpost','http://www.latimes.com/world/middleeast/rss2.0.xml']

def render(template, news):
    rtemplate = Environment(loader=BaseLoader()).from_string(template)
    title = 'Summarized News Feed'

    return rtemplate.render(news=news, title=title)

news_list = []


feed = FeedParser('http://rss.nytimes.com/services/xml/rss/nyt/World.xml')
news = feed.get_simple_news('nytimes')
for n in news:
    news_list.append(n)

feed = FeedParser('http://rss.cnn.com/rss/edition_world.rss')
news = feed.get_simple_news('cnn')
for n in news:
    news_list.append(n)

feed = FeedParser('http://feeds.washingtonpost.com/rss/rss_blogpost')
news = feed.get_simple_news('washingtonpost')
for n in news:
    news_list.append(n)


feed = FeedParser('http://www.latimes.com/world/middleeast/rss2.0.xml')
news = feed.get_simple_news('latimes')
for n in news:
    news_list.append(n)

all_news = []
for dic in news_list:
    content = re.sub('<.*?>', '', dic['content'])
    content = re.sub('\\t', '', content)
    content = re.sub('\\n', '', content)
    all_news.append(dic['title'])
    dic['no_tags']= content
    all_news.append(content)

doc = " ".join(all_news)
#tab = Table(language='english',ngram_range=(1,2), max_df=0.8)
#tab.init(all_news)
#summ = summary(doc, tab, 4, bm25=True)
#pr = PageRanker(tab, sim=0.2)
#PageRanker.init(tab, doc)
#ranked = pr.rank(prior_fn=PageRanker.prior_sentence_ix, weight_fn=PageRanker.weight_shared_noun_phrases,
#                 prior_calc=PageRanker.prior_quotient,
#                 prestige_calc=PageRanker.prestige_2_levels)
#summ = [s for s in sorted(ranked, key=ranked.get, reverse=True)[:5]]

clf = joblib.load('mlp.clf')
summ = classifier_summary(doc, 5, clf)

new_summ = []
for f in summ:
    done = False
    for new in news_list:
        title_sent = tokenize_sentences(new['title'])
        for s in title_sent:
            if s.startswith(f[:20]):
                new_summ.append(f + '\n' + '<a href=' + new['url'] + '>Extracted from ' + \
                    new['from'] + '</a>' + '</p>')

        title_cont = tokenize_sentences(new['no_tags'])
        for s in title_cont:
            if s.startswith(f[:20]):
                new_summ.append(f + '\n' + '<a href=' + new['url'] + '>Extracted from ' + \
                    new['from'] + '</a>' + '</p>')

with open('templates/index.html', 'r') as file:
    template = file.read()
news = {'title': 'RSS summary', 'content':" ".join(new_summ)}
with open('index.html', 'w+') as file:
    file.write(render(template, news))
