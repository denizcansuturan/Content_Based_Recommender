#############################
# Content Based Recommendation
#############################

#############################
# Recommendation Development Based on Film Overviews
#############################

# 1. Creating the TF-IDF Matrix
# 2. Creating Cosine Similarity Matrix
# 3. Making Recommendations Based on Similarities
# 4. Preparation of the Script

#################################
# 1. Creating the TF-IDF Matrix
#################################

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 500)
# pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape
df.columns
df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english")
# the words that are very common in languages and do not mean significant meaning
# are erased by specifying the language into TfidfVectorizer function
# in this case mentioned words are "the", "a", "and", "on" etc.

# tfidf is just an object

# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')
# NaN values could cause problems as if they were the overview of the movies
# so we turned the NaN values into just empty values

tfidf_matrix = tfidf.fit_transform(df['overview']).astype(np.float32)
# tfidf object is fitted(perform the operation) and
# transformed(values are replaced with new values permanently)
# .astype(np.float32) is used for smaller size

tfidf_matrix.shape
# (45466, 75827)
# 45466 rows are the movie descriptions
# 75827 columns are the unique words in all descriptions

df['overview'].shape

tfidf.get_feature_names_out()
# returns the unique names in the columns

tfidf_matrix.toarray()
# returns the scores

#################################
# 2. Creating Cosine Similarity Matrix
#################################

cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32),
                               tfidf_matrix.astype(np.float32))
# .astype(np.float32) is used for smaller size

cosine_sim.shape
cosine_sim[1]


#################################
# 3. Making Recommendations Based on Similarities
#################################

indices = pd.Series(df.index, index=df['title'])
# every movie and their indices are extracted
indices.head()

indices.index.value_counts().head()
# some movies are duplicated, extra records of movies must be eliminated.
# we need to take the last record so that the analysis would be more up to date

indices = indices[~indices.index.duplicated(keep='last')]
# default term is "first" we kept the last record

indices["Cinderella"]

indices["Sherlock Holmes"]

indices.index.value_counts().head()
# now every movie has 1 record

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]
# similarity scores between "Sherlock Holmes" and every other movie

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])
# more readable scores data frame for "Sherlock Holmes"

movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
# for recommendation purposes we just need the high similarity scores
# 10 movie indices which shows highest similarity with Sherlock Holmes are extracted
# movie in index 0 is not included since it is the "Sherlock Holmes" movie itself

df['title'].iloc[movie_indices]
# from the indices title of the movies that will be recommended to users are found

#################################
# 4. Preparation of the Script
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # creating indices
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # finding the index of mentioned title
    movie_index = indices[title]
    # calculating similarities
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # top 10 similar movies
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix.astype(np.float32), tfidf_matrix.astype(np.float32))
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)

