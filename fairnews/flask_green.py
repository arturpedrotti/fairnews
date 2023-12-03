from flask import Flask, render_template, request
import pandas as pd
from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.express as px
from transformers import pipeline
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
portuguese_stopwords = set(stopwords.words('portuguese'))

app = Flask(__name__)

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def get_news_data(search_term):
    try:
        googlenews = GoogleNews(lang='pt', region='BR')
        googlenews.search(search_term)
        results = googlenews.result()
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error fetching news data: {e}")
        return pd.DataFrame()

def add_sentiment_analysis(df):
    try:
        df['sentiment'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
        df['sentiment_score'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
        return df
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return df

def create_wordcloud(text):
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=portuguese_stopwords).generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as e:
        print(f"Error creating wordcloud: {e}")
        return None

def create_interactive_sentiment_graph(df):
    if 'sentiment_score' in df and 'media' in df and 'sentiment' in df:
        fig = px.scatter(df, x='sentiment_score', y='media', color='sentiment', hover_data=['title'])
        return fig.to_html(full_html=False)
    else:
        return "<p>Graph cannot be generated due to missing data.</p>"

@app.route('/', methods=['GET', 'POST'])
def index():
    news_data = pd.DataFrame()
    wordcloud_image = None
    sentiment_graph = None
    error_message = ''

    if request.method == 'POST':
        search_term = request.form.get('search_term', '')
        if search_term:
            news_data = get_news_data(search_term)
            if not news_data.empty:
                news_data = add_sentiment_analysis(news_data)
                wordcloud_image = create_wordcloud(' '.join(news_data['title']))
                sentiment_graph = create_interactive_sentiment_graph(news_data)
            else:
                error_message = "No news found for the search term."
        else:
            error_message = "Please enter a search term."

    return render_template('index.html', news_data=news_data.to_dict('records'), wordcloud_image=wordcloud_image, sentiment_graph=sentiment_graph, error_message=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

