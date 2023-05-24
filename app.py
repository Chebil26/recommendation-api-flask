from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

app = Flask(__name__)


def load_and_process_data():
    # read data frames
    static_dir = os.path.join(app.root_path, 'static')
    # read data frames
    books = pd.read_csv(os.path.join(static_dir, "Books.csv"), low_memory=False)
    users = pd.read_csv(os.path.join(static_dir, "Users.csv"), low_memory=False)
    ratings = pd.read_csv(os.path.join(static_dir, "Ratings.csv"), low_memory=False)
    

    # preprocess data frames
    books.drop_duplicates(inplace=True)
    users.drop("Age", axis=1, inplace=True)
    ratings.dropna(inplace=True)
    ratings.drop_duplicates(inplace=True)

    # merge data frames
    ratings_with_name = ratings.merge(books, on='ISBN')
    complete_df = ratings_with_name.merge(users, on='User-ID')

    # collaborative filtering-based recommendation system
    x = complete_df.groupby('User-ID').count()['Book-Rating'] > 200
    knowledgable_users = x[x].index
    filtered_rating = complete_df[complete_df['User-ID'].isin(knowledgable_users)]
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    similarity_score = cosine_similarity(pt)
    
    return books, pt, similarity_score


books, pt, similarity_score = load_and_process_data()
# recommendation function
def recommend(book_name):
    # Find the closest match to the given book name
    closest_match = process.extractOne(book_name, pt.index)[0]
    
    # Get the index of the closest match
    index = np.where(pt.index == closest_match)[0][0]
    
    # Get the most similar books based on the index
    similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x:x[1], reverse=True)[1:6]
    
    data = {}
    for i in similar_books:
        temp_df = books[books['Book-Title'] == pt.index[i[0]]].drop_duplicates('Book-Title')
        book_info = {
            'Book-Author': list(temp_df['Book-Author'].values)[0],
            'Image-URL-M': list(temp_df['Image-URL-M'].values)[0]
        }
        data[list(temp_df['Book-Title'].values)[0]] = book_info

    return data

@app.route('/recommendation', methods=['GET'])
def get_recommendation():
    book_name = request.args.get('book_name')
    recommendations = recommend(book_name)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run()