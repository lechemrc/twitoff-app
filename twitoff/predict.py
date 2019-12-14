"""Prediction of Users based on Tweet embeddings."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .models import User
from .twitter import BASILICA


def predict_user(user1_name, user2_name, tweet_text):
    """Determine and return which user is more likely to say a given Tweet.

    # Arguments
        user1_name: str, twitter user name for user1 in comparison
        user1_name: str, twitter user name for user2 in comparison
        tweet_text: str, tweet text to evaluate
    # Returns
        True if user1 has at least one tweet that's closer (measured in cosine similarity) to the sample tweet.
        False otherwise
    """
    user1 = User.query.filter(User.name == user1_name).one()
    user2 = User.query.filter(User.name == user2_name).one()
    user1_embeddings = np.array([tweet.embedding for tweet in user1.tweets])
    user2_embeddings = np.array([tweet.embedding for tweet in user2.tweets])
    user1_neigh = NearestNeighbors(metric='cosine').fit(user1_embeddings)
    user2_neigh = NearestNeighbors(metric='cosine').fit(user2_embeddings)
    tweet_embedding = BASILICA.embed_sentence(tweet_text, model='twitter')
    tweet_embedding = np.array(tweet_embedding).reshape(1, -1)
    user1_neigh_dist, _ = user1_neigh.kneighbors(X=tweet_embedding, n_neighbors=1)
    user2_neigh_dist, _ = user2_neigh.kneighbors(X=tweet_embedding, n_neighbors=1)
    user1_neigh_dist = user1_neigh_dist[0][0]
    user2_neigh_dist = user2_neigh_dist[0][0]

    return user1_neigh_dist < user2_neigh_dist