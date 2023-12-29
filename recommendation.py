import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

def preprocess_text(text):
    return text.lower().split()

def vectorize_text(model, text):
    words = preprocess_text(text)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def prepare_data(filepath, save_file='saved_objects_word2vec.pkl'):
    global word2vec_model, combined_features, df_courses

    if os.path.exists(save_file):
         with open(save_file, 'rb') as f:
            word2vec_model, combined_features, df_courses = pickle.load(f)
    else:
        df_courses = pd.read_csv(filepath)
        df_courses['text'] = df_courses['Course Name'] + ' ' + df_courses['Level']
        text_corpus = [preprocess_text(doc) for doc in df_courses['text']]
        word2vec_model = Word2Vec(sentences=text_corpus, vector_size=100, window=5, min_count=1, workers=4)
        df_courses['text_vector'] = df_courses['text'].apply(lambda x: vectorize_text(word2vec_model, x))
        df_courses['Rating'] = pd.to_numeric(df_courses['Rating'], errors='coerce').fillna(df_courses['Rating'].mean())
        scaler = MinMaxScaler()
        df_courses['Rating_scaled'] = scaler.fit_transform(df_courses[['Rating']])
        combined_features = np.hstack([np.vstack(df_courses['text_vector'].values), df_courses['Rating_scaled'].values.reshape(-1, 1)])
        with open(save_file, 'wb') as f:
            pickle.dump((word2vec_model, combined_features, df_courses), f)

def get_recommendations(user_input, level_filter=None, platform_filter=None):
    global word2vec_model, combined_features, df_courses

    # Filter courses based on level and platform
    if level_filter and level_filter.lower() != 'any':
        df_courses_filtered = df_courses[df_courses['Level'].str.lower() == level_filter.lower()]
    else:
        df_courses_filtered = df_courses.copy()

    if platform_filter and platform_filter.lower() != 'any':
        df_courses_filtered = df_courses_filtered[df_courses_filtered['Platform'].str.lower() == platform_filter.lower()]

    # Check if the length of the filtered DataFrame is greater than 0 before further operations
    if len(df_courses_filtered) > 0:
        # Calculate similarity scores based on the filtered DataFrame
        user_vector = vectorize_text(word2vec_model, user_input)
        combined_user_vector = np.hstack([user_vector, [0]])  # Assuming a zero rating for the user input
        similarity_scores = cosine_similarity([combined_user_vector], combined_features[df_courses_filtered.index]).flatten()
        df_courses_filtered['similarity'] = similarity_scores

        # Sort courses based on similarity scores and ratings
        df_sorted = df_courses_filtered.sort_values(by=['similarity', 'Rating'], ascending=[False, False])

        return df_sorted.head(10)[['Course Name', 'Level', 'Rating', 'Instructor/Institution', 'Platform', 'Course Link', 'Description']]
    else:
        # If there are no matching courses, return an empty DataFrame
        return pd.DataFrame(columns=['Course Name', 'Level', 'Rating', 'Instructor/Institution', 'Platform', 'Course Link', 'Description'])

filepath='modified_ratings_md2.csv'
prepare_data(filepath)

# # Example usage:
# user_input = "java"
# level_filter = "any"
# platform_filter = "udacity"
# recommendations = get_recommendations(user_input, level_filter, platform_filter)
# print(recommendations)
