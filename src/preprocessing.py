import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from io import BytesIO
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
from langdetect import detect

import tensorflow as tf
from joblib import Parallel, delayed
import re

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# get author names
def get_author_names(author_refs):
    authors = []
    for author_ref in author_refs:
        author_id = author_ref.get("author", {}).get("key", "").replace("/authors/", "")
        if author_id:
            author_url = f"https://openlibrary.org/authors/{author_id}.json"
            response = requests.get(author_url)
            if response.status_code == 200:
                author_data = response.json()
                authors.append(author_data.get("name", "Unknown"))
    return ", ".join(authors) if authors else "Unknown"

# get book details
def get_book_details(work_id):
    url = f"https://openlibrary.org/works/{work_id}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        title = data.get("title", "Unknown Title")
        author_refs = data.get("authors", [])
        authors = get_author_names(author_refs)
        
        publish_dt = data.get('first_publish_date', 'Unknown')
        
        cover_id = data.get('covers', [-1])[0]  # get first cover id or -1 if no covers available
        cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        
        description = data.get("description", {})
        if isinstance(description, dict):
            description = description.get("value", "No description available")
        elif isinstance(description, str):
            description = description.strip()
        else:
            description = "No description available"
        
        subjects = ", ".join(data.get("subjects", ["Unknown"]))
        
        return {
            "work_id": work_id,
            "title": title,
            "authors": authors,
            "description": description,
            "subjects": subjects,
            "cover_url": cover_url,
            "publish_date": publish_dt,
        }
    return None

# process book list
def process_book_list(book_list):
    all_book_data = []

    for book_id in book_list:
        work_id = book_id[0]
        book_details = get_book_details(work_id)
        time.sleep(1)  # increased sleep time to 5 seconds
        all_book_data.append(book_details)

    book_df = pd.DataFrame(all_book_data)
    
    book_df = book_df[~(book_df['cover_url'] == "https://covers.openlibrary.org/b/id/-1-L.jpg")]    
    return pd.DataFrame(all_book_data)

# get work IDs by genre
def get_work_ids_by_genre(genre="Adventure", sample_size=5, language="eng"):
    work_ids = []
    limit_per_page = min(sample_size, 100)
    
    url = f"https://openlibrary.org/search.json?subject={genre}&language={language}&limit={limit_per_page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        books = data.get("docs", [])
        for book in books:
            work_id = book.get("key", "").replace("/works/", "")
            first_publish_year = book.get("first_publish_year", "")
            
            if work_id:
                work_ids.append((work_id, first_publish_year))
            if len(work_ids) >= sample_size:
                break
    else:
        print(f"Failed to fetch books for genre: {genre} in language: {language}")
    return work_ids

# get image from URL
def get_img(cover_url, title, publish_date, display=True, delay=5):
    time.sleep(delay)
    response = requests.get(cover_url)
    if response.status_code == 200:
        img = mpimg.imread(BytesIO(response.content), format='jpg')
        if display==True:
            plt.imshow(img)
            plt.axis("off")
            plt.title(publish_date)
            plt.suptitle(title)
            plt.show()
        return img

# get multiple images
def get_imgs(df, display=True, delay=5):
    urls = df['cover_url']
    titles = df['title']
    publish_dates = df['publish_date']
    image_data = []
    
    for url, title, publish_date in zip(urls, titles, publish_dates):
        img_data = get_img(url, title, publish_date, display=display, delay=delay)
        image_data.append(img_data)

    return image_data

# preprocess image
def preprocess_img(image, max_height, max_width):
    img = image.copy()
    height, width, _ = img.shape
    
    pad_height = max_height - height if height < max_height else 0
    pad_width = max_width - width if width < max_width else 0
    
    mean_color = img.mean(axis=(0, 1))[2]  # calculate mean color of the image
    # print(img.shape, mean_color)
    
    padded_img = np.pad(img, 
                        ((0, pad_height),  # pad height (bottom)
                        (0, pad_width),    # pad width (right)
                        (0, 0)),           # not padding the rgb
                        mode='constant', 
                        constant_values=mean_color)  # padding with mean color
    
    normalized_img = padded_img / 255.0  # normalize rgb vals
    flattened_img = normalized_img.flatten() # .astype(np.float16)  # convert to float16
    return flattened_img

# Function to preprocess images with downsampling
def preprocess_images(images, downsample_factor=4):
    imgs = images.copy()
    
    # Downsample images
    downsampled_imgs = imgs.apply(lambda img: img[::downsample_factor, ::downsample_factor])
    
    max_height = max(downsampled_imgs.apply(lambda x: x.shape[0]))
    max_width = max(downsampled_imgs.apply(lambda x: x.shape[1]))
    
    processed_images = []
    
    for img in downsampled_imgs:
        processed_img = preprocess_img(img, max_height, max_width)
        processed_images.append(processed_img)
    
    return processed_images

# Function to pull and process cover images
def pull_and_process_cover_image(genres, sample_size=100, filename='covers'):
    print('Pulling book details and cover urls...')    
    covers_df = pd.DataFrame()
    
    for genre in genres:
        time.sleep(5) # reduced sleep time to 10 seconds
        print(genre)
        
        work_ids = get_work_ids_by_genre(genre=genre, sample_size=sample_size)
        genre_df = process_book_list(work_ids)
        genre_df['genre'] = genre
        covers_df = pd.concat([covers_df, genre_df], ignore_index=True)
        
    print('Pull successful, saved covers_df to pickle file')
    print(covers_df)

    covers_df = covers_df[~covers_df['cover_url'].str.contains('id/-1-L')] # drop rows with no cover url
    
    print('Pulling cover images from urls...')
    covers_df['unprocessed_img'] = get_imgs(covers_df, display=False, delay=2) # reduced delay to 2 seconds
    covers_df = covers_df[~covers_df['unprocessed_img'].isna()] # drop rows with no cover image
    
    print('Processing cover images...')
    covers_df['processed_img'] = preprocess_images(covers_df['unprocessed_img'])
    return covers_df

def plot_image(img_array, title=None):
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
    
def resize_image_tf(img, target_height, target_width):
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
    resized_img = tf.image.resize(img_tensor, [target_height, target_width], method=tf.image.ResizeMethod.BILINEAR)
    return resized_img.numpy()

def downsample_image(img, downsample_factor):
    return img[::downsample_factor, ::downsample_factor]

def preprocess_image(img, target_height, target_width, downsample_factor):
    resized_img = resize_image_tf(img, target_height, target_width)
    downsampled_img = downsample_image(resized_img, downsample_factor)
    normalized_img = downsampled_img / 255.0
    flattened_img = normalized_img.flatten()
    return flattened_img

def preprocess_images(images, target_height=500, target_width=350, downsample_factor=4, n_jobs=-1):
    imgs = images.copy().values
    
    # Parallel processing
    processed_imgs = Parallel(n_jobs=n_jobs)(
        delayed(preprocess_image)(img, target_height, target_width, downsample_factor) for img in imgs
    )
    return processed_imgs

def extract_years(dates):
    pattern = re.compile(r"\b(\d{4})\b")
    years = []
    
    for d in dates:
        match = pattern.search(d)
        if match:
            years.append(match.group(1))
        else:
            years.append(None)
    
    return years

def year_to_decade(year_str):
    if not year_str:
        return None
    try:
        yr = int(year_str)
        decade_start = (yr // 10) * 10
        return f"{decade_start}s"
    except ValueError:
        return None

def prep_data(genres = ["Fantasy"], sample_size = 5, filename='data'):
    try:
        covers_df = pd.read_pickle(f'{filename}.pkl')
        print('Loaded from pickle file')
    
    except:
        print('No pickle file found, pulling from api...')
        covers_df = pull_and_process_cover_image(genres, sample_size=sample_size) # sample size is per genre and pre filtering
        print('initial df shape:', covers_df.shape, '\n')
        covers_df.to_pickle(f'{filename}.pkl')
        print('Saved to pickle file')
    
    # added reprocessed imagees and image shapes since the added whitespace hurt the modeling
    covers_df['reprocessed_img'] = preprocess_images(covers_df['unprocessed_img'])
    covers_df['reprocessed_img_shape'] = covers_df['reprocessed_img'].apply(lambda x: x.shape)
    
    print('view processing example-')
    sample = covers_df.loc[0]
    plot_image(sample['unprocessed_img'])
    plot_image(sample['reprocessed_img'].reshape(125,88,3))
    
    covers_df['publish_yr'] = extract_years(covers_df['publish_date'].values)
    covers_df['publish_decade'] = covers_df['publish_yr'].apply(year_to_decade)
    
    covers_df = covers_df[~covers_df['publish_yr'].isna()]
    print('df shape post unknown date drop:', covers_df.shape, '\n')\
    
    return covers_df



# ### Example usage ###
# if __name__ == "__main__":
#     genres = [
#             "Fantasy", "Biography","Mystery", "Romance",
#             "Science Fiction","Thriller", "Horror","Non-Fiction",
#             "Self-Help","True Crime","Young Adult"
#             ]    
#     covers_df = prep_data(genres, sample_size=5) # sample size is per genre
#     # covers_df.to_pickle(f'{filename}.pkl')