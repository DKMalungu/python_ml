"""
The dataset can be downloaded from: https://grouplens.org/datasets/movielens/100k/
"""

# Importing Libraries
import pandas as pd
import numpy as np

# Step 1: Loading the dataset into python
# Loading users file
"""
This information is from the read me doc of the dataset
u.user     -- Demographic information about the users; this is a tab
              separated list of
              user id | age | gender | occupation | zip code
              The user ids are the ones used in the u.data data set."""
# User data column names
user_columns = ["user_id", 'age', 'gender', 'occupation', 'zip_code']
df_user = pd.read_csv(filepath_or_buffer='./data_store/ml-100k/u.user', sep='|', names=user_columns,
                      encoding='latin-1')

# Loading ratings file:
"""
This information is from the read me doc of the dataset
u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
              Each user has rated at least 20 movies.  Users and items are
              numbered consecutively from 1.  The data is randomly
              ordered. This is a tab separated list of 
              user id | item id | rating | timestamp. 
              The time stamps are unix seconds since 1/1/1970 UTC   """

rating_columns = ['user_id', 'item_id', 'rating', 'timestamp']
df_rating = pd.read_csv(filepath_or_buffer='./data_store/ml-100k/u.data', sep='\t', names=rating_columns,
                        encoding='latin-1')

# loading items file:
"""
u.item     -- Information about the items (movies); this is a tab separated
              list of
              movie id | movie title | release date | video release date |
              IMDb URL | unknown | Action | Adventure | Animation |
              Children's | Comedy | Crime | Documentary | Drama | Fantasy |
              Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
              Thriller | War | Western |
              The last 19 fields are the genres, a 1 indicates the movie
              is of that genre, a 0 indicates it is not; movies can be in
              several genres at once.
              The movie ids are the ones used in the u.data data set.
"""
items_columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url', 'unknown', 'action', 'adventure', 'animation',
                 'children', 'comedy', 'crime', 'documentary', 'drama', 'fantasy',
                 'film_noir', 'horror', 'musical', 'mystery', 'romance', 'sci_fi',
                 'thriller', 'war', 'western']
df_items = pd.read_csv(filepath_or_buffer='./data_store/ml-100k/u.item', sep='|', names=items_columns,
                       encoding='latin-1')

# Step 2: Descriptive Analysis of the data in each file
# user data file
print('Shape of user data file', df_user.shape)
print("Data types of the different columns in user data fil: ", df_user.dtypes)
print("A sample of the data in user dat file: ", df_user.sample(5))

"""Results:
a)Shape
Shape of user data file (943, 5) -  The user file data is made up of 943 rows and 5 columns

b) Data types
Data types of the different columns in user data fil:  
user_id        int64
age            int64
gender        object
occupation    object
zip_code      object
dtype: object - The dataset is mad up of data of the type object and int64

c) Sample data
A sample of the data in user dat file:       
user_id  age gender occupation zip_code
743      744   35      M  marketing    47024
292      293   24      M     writer    60804
460      461   15      M    student    98102
897      898   23      M  homemaker    61755
582      583   44      M   engineer    29631
The above is a sample of five random rows from the dataset
"""

# rating data file
print('Shape of user data file', df_rating.shape)
print("Data types of the different columns in user data fil: ", df_rating.dtypes)
print("A sample of the data in user dat file: ", df_rating.sample(5))
"""Results
a) Shape
Shape of user data file (100000, 4) - the dataset is made up of hundred thousand rows and four columns

b) Dtypes
Data types of the different columns in user data fil:  user id       object
item_id      float64
rating       float64
timestamp    float64
dtype: object - it made up of the float data type
A sample of the data in user dat file:                         user id  item_id  rating  timestamp
37410   561\t410\t1\t885810117      NaN     NaN        NaN
93150  314\t1297\t4\t877890734      NaN     NaN        NaN
39493  378\t1531\t4\t880056423      NaN     NaN        NaN
69012    89\t301\t5\t879461219      NaN     NaN        NaN
7593     83\t575\t4\t880309339      NaN     NaN        NaN
The above is a sample of five random rows from the dataset
"""

# Items data file
print('Shape of user data file', df_items.shape)
print("Data types of the different columns in user data fil: ", df_items.dtypes)
print("A sample of the data in user dat file: ", df_items.sample(5))

# Step 3: Building a collaborative filtering model from scratch
# Finding the number of unique users and movies
unique_users = df_rating['user_id'].unique().shape[0]
unique_item = df_rating['item_id'].unique().shape[0]

# Create a user_item matrix which can be used to calculate the similarty between users and items
data_matrix = np.zeros((unique_users, unique_item))

for line in df_rating.itertuples():
    data_matrix[line[1] - 1, line[2] - 1] = line[3]

# Calculating the similarity of user rating and items
from sklearn.metrics.pairwise import pairwise_distances

user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')


# Creating the prediction function
def predict(rating, similarity, prediction_type='user'):
    prediction_type = prediction_type.lower()
    if prediction_type == 'user':
        mean_user_rating = rating.mean(axis=1).reshape(-1, 1)

        rating_diff = (rating - mean_user_rating)

        pred = mean_user_rating + similarity.dot(rating_diff) / np.array([np.abs(similarity).sum(axis=1)]).T

    elif prediction_type == 'item':
        pred = rating.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    else:
        raise ValueError(f"The prediction type provided: {prediction_type} is not in options ->> ('item' or 'user')")


# Finally making prediction
user_prediction = predict(rating=data_matrix, similarity=user_similarity, prediction_type='user')
item_prediction = predict(rating=data_matrix, similarity=item_similarity, prediction_type='item')

