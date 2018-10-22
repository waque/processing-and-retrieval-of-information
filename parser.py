import feedparser
from jinja2 import Environment, BaseLoader

class FeedParser:
    def __init__(self, url, lazy_load=False):
        self.url = url
        self.feed = None

        if not lazy_load:
            self.parse()

    def parse(self):
        self.feed = feedparser.parse(self.url)
        # for key in self.feed['entries'][0]:
            # print(key)
        # print(self.feed['entries'][0]['content'][0]['value'])


    def get_news(self):
        return self.feed['entries']

    def get_news_titles(self):
        news = self.get_news()
        return [n['title'] for n in news]

    def get_simple_news(self, website):
        news = self.get_news()
        simplified = []
        for n in news:
            if (website == 'nytimes'):
                new = {'title': n['title']}
                content = n.get('content', None)
                if content is not None:
                    new['content'] = n['content'][0]['value']
                    new['url'] = n['link']
                    new['from'] = 'nytimes'
                    simplified.append(new)
            if (website == 'washingtonpost'):
                new = {'title': n['title'], 'content': n['summary'], 'url': n['link'], 'from': 'washingtonpost'}
                simplified.append(new)
            if (website == 'cnn'):
                new = {'title': n['title'], 'content': n['summary_detail']['value'], 'url': n['link'], 'from': 'cnn'}
                simplified.append(new)
            if (website == 'latimes'):
                new = {'title': n['title'], 'content': n['summary_detail']['value'], 'url': n['link'], 'from': 'latimes'}
                simplified.append(new)
        return simplified

    def get_title(self):
        return self.feed['feed']['title']
