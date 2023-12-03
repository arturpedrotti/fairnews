from flask import Flask, render_template, request
import pandas as pd
from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import nltk
from nltk.corpus import stopwords
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)
portuguese_stopwords = set(stopwords.words('portuguese'))

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Function to fetch news data
def get_news_data(search_term):
    googlenews = GoogleNews(lang='pt', region='BR')
    googlenews.search(search_term)
    results = googlenews.result()
    return pd.DataFrame(results)

# Function for sentiment analysis
def add_sentiment_analysis(df):
    df['sentiment'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['sentiment_score'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    return df

# Function to create a word cloud image
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=portuguese_stopwords).generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')

# Route for the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    news_data = pd.DataFrame()
    wordcloud_image = None 
    error_message = ''
    
    if request.method == 'POST':
        search_term = request.form.get('search_term', '')
        if search_term:
            news_data = get_news_data(search_term)
            if not news_data.empty:
                news_data = add_sentiment_analysis(news_data)
                wordcloud_image = create_wordcloud(' '.join(news_data['title']))
            else:
                error_message = "No news found for the search term."
        else:
            error_message = "Please enter a search term."

    return render_template('index.html', news_data=news_data.to_dict('records'), wordcloud_image=wordcloud_image, error_message=error_message)

# Main function to run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
