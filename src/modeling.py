import sys
import os

sys.path.append(os.path.dirname(__file__))
# print(sys.path)

import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import pandas  as pd
import numpy as np
import re

from concurrent.futures import ThreadPoolExecutor
from src.preprocessing import *

from sklearn.manifold import TSNE
from umap import UMAP

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.sparse import hstack
from sklearn.metrics import adjusted_rand_score
from sklearn.naive_bayes import GaussianNB
import warnings
# Load the SentenceTransformer model.
from sentence_transformers import SentenceTransformer
from datetime import date

warnings.filterwarnings("ignore", message="...", category=FutureWarning)
warnings.filterwarnings("ignore", message="...", category=UserWarning)

def img_nlp_model(modeled_df, 
                    seed,
                    text_approach='merged', 
                    params={
                        'clustering_algo': GaussianNB, 
                        'clustering_params': {}, 
                        'metric': 'cosine', 
                        'min_dist': 0.01, 
                        'n_neighbors': 30,
                        'n_components_text': 2,         # For merged approach
                        'n_components_img': 2,          # For image UMAP
                        'n_components_title': 2,        # For separated title embeddings
                        'n_components_description': 2   # For separated description embeddings
                    },
                    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")):
    """
    Processes text using a pre-trained SentenceTransformer model and combines them with image features for clustering.
    
    Parameters:
        - modeled_df: DataFrame with columns 'title', 'description', 'reprocessed_img', 'genre'
        - seed: a seed for reproducibility (if needed)
        - text_approach: 'merged' to combine title and description; otherwise, processes them separately.
        - params: dictionary of parameters for UMAP and clustering.
        - embedding_model: a pre-trained SentenceTransformer model.
    
    Returns:
        A tuple containing the top-3 genre predictions (both names and indices), top genre prediction, 
        and the combined embedding (used for clustering).
    """
    
    modeled_df['description'] = modeled_df['description'].apply(lambda x: x.replace('No description available', ''))    
    
    # Helper function: For SentenceTransformer, simply encode the full text
    def get_document_embedding(text, model):
        return model.encode(text)
    
    if text_approach == 'merged':
        # Merge title and description, preprocess, then compute the document embedding
        modeled_df['text'] = (modeled_df['title'] + ' ' + modeled_df['description']).apply(preprocess_text)
        X_text = np.vstack(modeled_df['text'].apply(lambda t: get_document_embedding(t, embedding_model)).values)
        
        # Apply UMAP on the merged text embeddings
        umap_text = UMAP(
            n_components=params.get('n_components_text', 2),
            n_neighbors=params.get('n_neighbors', 15),
            min_dist=params.get('min_dist', 0.1),
            metric=params.get('metric', 'cosine'),
        )
        X_text_umap = umap_text.fit_transform(X_text)
    else:
        # Process title and description separately
        modeled_df['title'] = modeled_df['title'].apply(lambda x: preprocess_text(x))
        modeled_df['description'] = modeled_df['description'].apply(lambda x: preprocess_text(x))
        
        # Compute the document embedding for title
        X_title = np.vstack(modeled_df['title'].apply(lambda t: get_document_embedding(t, embedding_model)).values)
        # Compute the document embedding for description
        X_description = np.vstack(modeled_df['description'].apply(lambda t: get_document_embedding(t, embedding_model)).values)
        
        # Apply UMAP on title embeddings
        umap_title = UMAP(
            n_components=params.get('n_components_title', 2),
            n_neighbors=params.get('n_neighbors', 15),
            min_dist=params.get('min_dist', 0.1),
            metric=params.get('metric', 'cosine'),
        )
        X_title_umap = umap_title.fit_transform(X_title)
        
        # Apply UMAP on description embeddings
        umap_description = UMAP(
            n_components=params.get('n_components_description', 2),
            n_neighbors=params.get('n_neighbors', 15),
            min_dist=params.get('min_dist', 0.1),
            metric=params.get('metric', 'cosine'),
        )
        X_description_umap = umap_description.fit_transform(X_description)
        
        # Concatenate the title and description UMAP embeddings
        X_text_umap = np.hstack((X_title_umap, X_description_umap))
    
    # Process image data separately (assumes images are already preprocessed)
    X_img = np.stack(modeled_df['reprocessed_img'].values)
    umap_img = UMAP(
        n_components=params.get('n_components_img', 2),
        n_neighbors=params.get('n_neighbors', 15),
        min_dist=params.get('min_dist', 0.1),
        metric=params.get('metric', 'cosine'),
    )
    X_img_umap = umap_img.fit_transform(X_img)
    
    # Combine the image and text embeddings
    X_combined = np.hstack((X_img_umap, X_text_umap))
    
    # Clustering on the combined embeddings
    clustering_algo = params['clustering_algo']
    clustering_params = params['clustering_params']
    clustering_model = clustering_algo(**clustering_params)
    clustering_model.fit(X_combined, modeled_df['genre'].astype('category').cat.codes)
    
    # Predict probabilities for each genre
    genre_probs = clustering_model.predict_proba(X_combined)
    
    # Get the top 3 genre predictions for each book
    top_3_genres = np.argsort(genre_probs, axis=1)[:, -3:]
    top_3_genre_names = np.array(modeled_df['genre'].astype('category').cat.categories)[top_3_genres]
    modeled_df['top_3_genres'] = [list(genres) for genres in top_3_genre_names]
    
    # Get the top genre prediction for each book
    top_genre = np.argmax(genre_probs, axis=1)
    top_genre_names = np.array(modeled_df['genre'].astype('category').cat.categories)[top_genre]
    modeled_df['top_genre'] = top_genre_names
    
    return (modeled_df['top_3_genres'], 
            top_3_genres.tolist(), 
            modeled_df['top_genre'], 
            top_genre, 
            X_combined)

def calculate_ari(true_labels, cluster_labels):
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    return ari_score

def pull_model(seed, filepath):
    today = date.today()
    
    if filepath is None:
        filepath = f'data/data_{today}'
    
    try:
        modeled_df = pd.read_pickle(f'{filepath}_modeled.pkl')
        print("Data already modeled. Loading from pickle...")
    
    except FileNotFoundError:
        print('No modeled data found, modeling now...')
        
        # Define the genres and sample size
        genres = [
            "Fantasy", "Biography","Mystery", "Romance",
            "Science Fiction","Thriller", "Horror","Non-Fiction",
            "Self-Help","True Crime","Young Adult"
            ]  
        
        # Example usage of the prep_data function
        print("Preparing data...")
        covers_df = prep_data(genres, 300, filepath)
        
        covers_df = covers_df.drop_duplicates(subset='title')
        params = {'clustering_algo': GaussianNB, 'clustering_params': {}, 'metric': 'cosine', 'min_dist': 0.01, 'n_components': 2, 'n_neighbors': 30}
        
        # Filtering to English books
        covers_df = covers_df[covers_df['description'].apply(is_english)]
        
        modeled_df = covers_df.copy()
        print('Modeling...')
        modeled_df['top_3_genres_partitioned_txt'], modeled_df['top_3_genres_partitioned'], modeled_df['partitioned_pred_txt'], modeled_df['partitioned_pred'], partition_X_umap = img_nlp_model(modeled_df, seed, text_approach='partitioned', params=params)
        modeled_df['top_3_genres_merged_txt'], modeled_df['top_3_genres_merged'], modeled_df['merged_pred_txt'], modeled_df['merged_pred'], merged_X_umap = img_nlp_model(modeled_df, seed, params=params)
        
        true_labels = modeled_df['genre'].astype('category').cat.codes
        part_cluster_labels = modeled_df['partitioned_pred']
        merged_cluster_labels = modeled_df['merged_pred']
        
        ari_score = calculate_ari(true_labels, part_cluster_labels)
        print(f"Partitioned ARI: {ari_score}")
        ari_score = calculate_ari(true_labels, merged_cluster_labels)
        print(f"Merged ARI: {ari_score}")
        
        print("Saving data...")
        modeled_df.to_pickle(f'{filepath}_modeled.pkl')
    return modeled_df

# ### Example usage ###
# if __name__ == "__main__":
#     seed = 0
#     pull_model(seed) # if you have pulled unmodeled data, you can pass the filepath without .pkl to read that in


