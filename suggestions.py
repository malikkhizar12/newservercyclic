import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    # Add any additional text preprocessing steps here
    return text.lower()

def get_suggestions(search_words=None):
    filepath = 'modified_ratings_md2.csv'
    df = pd.read_csv(filepath)

    # If there are search words, filter courses based on search words
    if search_words:
        # Convert search_words to lowercase for case-insensitive matching
        search_words_lower = [word.lower() for word in search_words]

        # Preprocess course names and descriptions
        df['processed_name'] = df['Course Name'].apply(preprocess_text)
        df['processed_description'] = df['Description'].apply(preprocess_text)

        # Combine processed name and description
        df['combined_text'] = df['processed_name'] + ' ' + df['processed_description']

        # Vectorize the combined text using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

        # Vectorize the search words using TF-IDF
        search_tfidf = vectorizer.transform([' '.join(search_words_lower)])

        # Calculate cosine similarity between search words and courses
        similarity_scores = cosine_similarity(search_tfidf, tfidf_matrix).flatten()

        # Add a new column 'similarity_score' to the DataFrame
        df['similarity_score'] = similarity_scores

        # Sort courses based on similarity scores
        sorted_courses = df.sort_values(by='similarity_score', ascending=False)

        # Return the top 6 suggestions
        return sorted_courses.head(6)

    # If there are no search words, return a random sample of 6 courses
    return df.sample(n=6 )

# suggestions = get_suggestions(["machine", "learning", "python"])
# print(suggestions[['Course Name', 'similarity_score']])
