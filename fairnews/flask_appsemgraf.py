from flask import Flask, render_template, request
import pandas as pd
from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from transformers import pipeline

app = Flask(__name__)

# Inicializa a pipeline de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

def get_news_data(search_term):
    googlenews = GoogleNews(lang='pt', region='BR')
    googlenews.search(search_term)
    results = googlenews.result()
    return pd.DataFrame(results)

def add_sentiment_analysis(df):
    df['sentimento'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    return df

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf8')

def sentiment_to_color(sentiment):
    return {
        'positivo': 'green',
        'negativo': 'red',
        'neutro': 'grey'
    }.get(sentiment, 'black')  # Cor padrão

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
                news_data['color'] = news_data['sentimento'].apply(sentiment_to_color)
            else:
                error_message = "Nenhuma notícia encontrada para o termo de pesquisa."
        else:
            error_message = "Por favor, insira um termo de pesquisa."

    return render_template('index.html', 
                           news_data=news_data.to_dict('records'), 
                           wordcloud_image=wordcloud_image, 
                           error_message=error_message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

