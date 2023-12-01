import streamlit as st
import pandas as pd
from GoogleNews import GoogleNews
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from transformers import pipeline
import altair as alt

# Baixar stopwords do NLTK
nltk.download('stopwords')
portuguese_stopwords = set(stopwords.words('portuguese'))

# Lista de fontes de notícias mais relevantes
relevant_sources = ['jovem pan', 'bbc', 'cnn', 'g1', 'uol', 'veja', 'globo']

# Inicializa a pipeline de sentimentos
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Função para buscar e processar dados de notícias
def get_news_data(search_term):
    googlenews = GoogleNews(lang='pt', region='BR')
    googlenews.search(search_term)
    results = googlenews.result()
    df = pd.DataFrame(results)

    # Filtrar e ordenar notícias
    df['media_lower'] = df['media'].str.lower()
    df['relevance'] = df['media_lower'].apply(lambda x: x in relevant_sources)
    sorted_df = df.sort_values(by='relevance', ascending=False)

    return sorted_df

# Função para análise de sentimento
def add_sentiment_analysis(df):
    df['sentiment'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['label'])
    df['sentiment_score'] = df['title'].apply(lambda x: sentiment_pipeline(x)[0]['score'])
    return df

# Função para criar uma nuvem de palavras
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=portuguese_stopwords).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Função para criar um gráfico de sentimentos
def create_sentiment_chart(df):
    df = df.reset_index()
    color_scale = alt.Scale(domain=['positive', 'neutral', 'negative'], range=['green', 'grey', 'red'])
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X('index:O', axis=alt.Axis(labelAngle=0)),
        y='sentiment_score:Q',
        color=alt.Color('sentiment:N', scale=color_scale),
        tooltip=['index', 'title', 'sentiment_score']
    ).properties(
        width=700,
        height=400
    ).interactive()
    return chart

# Aplicativo Streamlit
def main():
    st.title("Análise e Visualização de Notícias")
    search_term = st.text_input("Digite um tema para buscar notícias:")

    if st.button('Buscar'):
        with st.spinner('Buscando notícias...'):
            news_df = get_news_data(search_term)
            news_df = news_df[news_df['title'].str.strip().astype(bool)]  # Remove linhas com títulos vazios
            if not news_df.empty:
                news_df = add_sentiment_analysis(news_df)
                fig_wordcloud = create_wordcloud(' '.join(news_df['title']))
                
                sentiment_chart = create_sentiment_chart(news_df)
                st.subheader("Gráfico de Sentimentos")
                st.altair_chart(sentiment_chart, use_container_width=True)

                st.subheader("Notícias Encontradas (Ordenadas por Relevância)")
                for index, row in news_df.iterrows():
                    st.text(f"{index + 1}. {row['title']} - Sentimento: {row['sentiment']} - Fonte: {row.get('media', 'Desconhecido')}")

                st.subheader("Wordcloud")
                st.pyplot(fig_wordcloud)
            else:
                st.error("Nenhuma notícia encontrada para o termo de pesquisa.")

if __name__ == "__main__":
    main()

