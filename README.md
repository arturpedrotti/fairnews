# 🌍 Fair News

Fair News is a Flask web application that analyzes the sentiment of news articles and presents the results in a visually interactive way. It generates word clouds from news headlines and plots the sentiment (positive, neutral, or negative) using dynamic graphs.

![Logo](static/img/Logo-fairnews.png)

---

## 🚀 Features

- 🔍 **Search News**: Search articles by keyword using GoogleNews.
- 💬 **Sentiment Analysis**: Classifies news as Positive, Neutral, or Negative using `transformers`.
- ☁️ **Word Cloud**: Displays a word cloud from the news titles.
- 📊 **Interactive Visualization**: Displays a sentiment scatterplot (via Plotly).

---

## 🛠 Tech Stack

- **Flask**
- **Pandas**
- **GoogleNews**
- **NLTK**
- **Transformers (HuggingFace)**
- **Matplotlib**, **Plotly**, **WordCloud**
- **HTML/CSS/JS**

---

## 🧪 Installation

```bash
git clone https://github.com/arturpedrotti/fairnews.git
cd fairnews
pip install -r requirements.txt
python3 app.py
```

---

## 👨‍💻 Author

Developed entirely by [Artur Pedrotti](https://github.com/arturpedrotti)  
Academic project under guidance of [Matheus Pestana](https://github.com/mateuspestana)

---

## 📝 License

MIT License. See [LICENSE.txt](LICENSE.txt) for details.
