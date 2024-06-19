import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from flask import Flask, request, jsonify, render_template
from googleapiclient.discovery import build

# Load the dataset
recipes = pd.read_csv('Cleaned_Indian_Food_Dataset.csv')

# Basic cleaning and preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_ingredients(ingredients):
    ingredients = ingredients.lower()
    ingredients = re.sub(r'\d+', '', ingredients)
    ingredients = re.sub(r'[^\w\s]', '', ingredients)
    tokens = ingredients.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

recipes['Cleaned-Ingredients'] = recipes['TranslatedIngredients'].apply(preprocess_ingredients)
recipes['Ingredient-count'] = recipes['TranslatedIngredients'].apply(lambda x: len(x.split(',')))

# Vectorize the cleaned ingredients
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(recipes['Cleaned-Ingredients'])

# Recommendation Function
def recommend_recipes(ingredients_list, top_n=1):
    ingredients_str = ' '.join(ingredients_list).lower()
    ingredients_str = preprocess_ingredients(ingredients_str)
    query_vec = vectorizer.transform([ingredients_str])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    recommended_indices = similarity_scores.argsort()[-top_n:][::-1]
    return recipes.iloc[recommended_indices]

# YouTube API function
def get_youtube_video(query):
    api_key = "AIzaSyAlNPSn3H_FWste4tsMVyxZmuSYGjxry0E"  # Replace with your actual YouTube API key
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(q=query, part='snippet', type='video', maxResults=1)
    response = request.execute()
    video_id = response['items'][0]['id']['videoId']
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    print(video_url)
    return video_url

# Flask App
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    ingredients = data.get('ingredients')
    if not ingredients:
        return jsonify({'error': 'No ingredients provided'}), 400
    recommendations = recommend_recipes(ingredients)
    recommendations_list = recommendations[['TranslatedRecipeName', 'TranslatedIngredients', 'TotalTimeInMins', 'Cuisine', 'TranslatedInstructions']].to_dict(orient='records')
    
    # Get YouTube video for the first recommended recipe
    video_url = get_youtube_video(recommendations.iloc[0]['TranslatedRecipeName'])
    
    return jsonify({'recommendations': recommendations_list, 'video_url': video_url})

if __name__ == '__main__':
    app.run(debug=True)
