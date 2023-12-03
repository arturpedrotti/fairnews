from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import base64
from io import BytesIO

nltk.download('stopwords')
portuguese_stopwords = set(stopwords.words('portuguese'))

app = Flask(__name__)

# Initialize your sentiment pipeline here
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        search_term = request.form.get('search_term')
        if search_term:
            news_df = get_news_data(search_term)
            news_df = add_sentiment_analysis(news_df)
            wordcloud_img = create_wordcloud(news_df)
            # Since we cannot directly use Altair in Flask as in Streamlit, we skip the chart generation.
            # You might want to use a different library to generate charts or serve them as static images.
            return render_template('index.html', news_data=news_df.to_dict('records'), wordcloud_image=wordcloud_img)
    return render_template('index.html')

def get_news_data(search_term):
    googlenews = GoogleNews(lang='pt', region='BR')
    googlenews.search(search_term)
    results = googlenews.result()
    return pd.DataFrame(results)

def add_sentiment_analysis(df):
    df['sentiment'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['sentiment_score'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    return df

def create_wordcloud(df):
    text = ' '.join(df['title'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=portuguese_stopwords).generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    wordcloud_img = base64.b64encode(img.getvalue()).decode('utf-8')
    return wordcloud_img

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

