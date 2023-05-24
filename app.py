from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)

static_dir = os.path.join(app.root_path, 'static')
books_filepath = os.path.join(static_dir, "Books.csv")
users_filepath = os.path.join(static_dir, "Users.csv")
ratings_filepath = os.path.join(static_dir, "Ratings.csv")

books = None
pt = None
similarity_score = None


def load_and_process_data():
    global books, pt, similarity_score
    
    # Read data frames
    books = pd.read_csv(books_filepath, usecols=['ISBN', 'Book-Title', 'Book-Author', 'Image-URL-M'], low_memory=False)
    users = pd.read_csv(users_filepath, usecols=['User-ID'], low_memory=False)
    ratings = pd.read_csv(ratings_filepath, usecols=['User-ID', 'ISBN', 'Book-Rating'], low_memory=False)
    
    # Preprocess data frames
    books.drop_duplicates(inplace=True)
    ratings.dropna(inplace=True)
    ratings.drop_duplicates(inplace=True)

    # Merge data frames
    ratings_with_name = ratings.merge(books, on='ISBN')
    complete_df = ratings_with_name.merge(users, on='User-ID')

    # Collaborative filtering-based recommendation system
    x = complete_df['User-ID'].value_counts() > 200
    knowledgable_users = x[x].index
    filtered_rating = complete_df[complete_df['User-ID'].isin(knowledgable_users)]
    y = filtered_rating['Book-Title'].value_counts() >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating', fill_value=0)
    similarity_score = cosine_similarity(pt)

    return


def recommend(book_name):
    # Find the closest match to the given book name
    closest_match = process.extractOne(book_name, pt.index)[0]
    
    # Get the index of the closest match
    index = np.where(pt.index == closest_match)[0][0]
    
    # Get the most similar books based on the index
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    data = {}
    for i in similar_books:
        temp_df = books.loc[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        book_info = {
            'Book-Author': temp_df['Book-Author'].values[0],
            'Image-URL-M': temp_df['Image-URL-M'].values[0]
        }
        data[temp_df['Book-Title'].values[0]] = book_info

    return data


@app.route('/recommendation', methods=['GET'])
def get_recommendation():
    book_name = request.args.get('book_name')
    recommendations = recommend(book_name)
    return jsonify(recommendations)


@app.route('/', methods=['GET'])
def home():
    return "Hello there!"


if __name__ == '__main__':
    load_and_process_data()
    app.run(debug=True, port=os.getenv("PORT", default=5000))
