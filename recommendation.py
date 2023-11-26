import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack
import pickle
import os

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['text'] = df['Course Name'] + ' ' + df['Course Description'] + ' ' + df['Skills']
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

    df['Course Rating'] = pd.to_numeric(df['Course Rating'], errors='coerce')
    mean_rating = df['Course3 Rating'].mean()
    df['Course Rating'].fillna(mean_rating, inplace=True)
    
    scaler = MinMaxScaler()
    df['Course Rating_scaled'] = scaler.fit_transform(df[['Course Rating']])
    rating_matrix = csr_matrix(df['Course Rating_scaled'].values).T

    combined_matrix = hstack([tfidf_matrix, rating_matrix])

    with open('saved_objects.pkl', 'wb') as f:
        pickle.dump((tfidf_vectorizer, combined_matrix, df), f)

def load_or_prepare_data(filepath):
    if not os.path.exists('saved_objects.pkl'):
        prepare_data(filepath)

    with open('saved_objects.pkl', 'rb') as f:
        return pickle.load(f)

def get_recommendations(user_input):
    filepath='Coursera_2.csv'
    tfidf_vectorizer, combined_matrix, df = load_or_prepare_data(filepath)
    
    user_vector = tfidf_vectorizer.transform([user_input])
    zero_rating_vector = csr_matrix((1, 1), dtype=float)
    combined_user_vector = hstack([user_vector, zero_rating_vector])
    
    similarity_scores = cosine_similarity(combined_user_vector, combined_matrix).flatten()
    similar_courses = similarity_scores.argsort()[-10:][::-1]
    
    recommended_courses = df.iloc[similar_courses]
    return recommended_courses
