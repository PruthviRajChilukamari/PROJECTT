from flask import Flask, render_template, flash, request, url_for, redirect, session
import os
import string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

IMAGE_FOLDER = os.path.join('static', 'img_pool')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")


@app.route('/sentiment_analysis_prediction', methods=['POST', "GET"])
def sent_anly_prediction():
    global probability, sentiment, text, img_filename
    if request.method == 'POST':
        text = request.form['text']
        sentiment = ''
        lower_case = text.lower()
        cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
        score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
        probability = 0.9
        if score['neg'] > score['pos']:
            sentiment = 'Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
        elif score['neg'] < score['pos']:
            sentiment = 'Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
        else:
            sentiment = 'Neutral'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Neutral_Emoji.png')

        # Using word_tokenize because it's faster than split()
        tokenized_words = word_tokenize(cleaned_text, "english")

        # Removing Stop Words
        final_words = []
        for word in tokenized_words:
            if word not in stopwords.words('english'):
                final_words.append(word)

        # Lemmatization - From plural to single + Base form of a word (example better-> good)
        lemma_words = []
        for word in final_words:
            word = WordNetLemmatizer().lemmatize(word)
            lemma_words.append(word)

        emotion_list = []
        with open('emotions.txt', 'r') as file:
            for line in file:
                clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
                word, emotion = clear_line.split(':')

                if word in lemma_words:
                    emotion_list.append(emotion)

    return render_template('home.html', text=text, sentiment=sentiment, probability=probability, image=img_filename)


if __name__ == "__main__":
    app.run()
