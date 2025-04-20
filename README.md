# The Evolving Face of Genre Through Time
### ISYE 6740 Class Project Spring 2025: Jeff Triemer
This project aimed to cluster books by their marketing (cover, title and description) as a prediction of genre, in order to measure the models effectiveness over time and then apply some simple regression to use the clustering error as a proxy to understand how effective or precise authors and publishers have become in utilizing these marketing tools over time. Note there is much room for improvement in this model. For one, a given book may have multiple genres, but for this project we will assume the most likely cluster/genre is the prediction, and the genre the book first appeared for when searching in OpenLibrary is the only 'true' genre. This is of course not wholly true, but will work as an assumption to test our hypothesis that marketing by genre has become more consise or similar over time.

##### Future improvements
- Improving the measure of clustering accuracy to account for multiple genres or sub-genres
    - Potentially applying something like WCSS to measure error not by missed prediction but error magnitude
- This model's image processing adds very little value to the overall prediction and would benefit massively from feature extraction
- Genre imbalance also contributes heavily to the model's confusion and could be treated
- The Genres used in this model could be aggregated or grouped in a more clear fashion for modeling, like grouping Sci-Fi/Fantasy or Mystery/Thriller
    - Similar to this, I have noted that the model may benefit from some general intelligence, currently the model may identify a title like 'Prince Caspian' as indicative of a Biography rather than a Fantasy as there is no such person (this of course adds a lot of resources to correct)
- Computationally the data pull could be massively improved my minimizing the imputed sleep times for the OpenLibrary API, as well as dropping any duplicates or books without clear published dates earlier in processing


# Setup
Navigate to requirements.txt and create environment to run the project using:
```
conda create --name project-env --file requirements.txt
```
Note: this step will take a while to for all the relevant nlp tools


Use this to enable its use in the results jupyter notebook:
```
conda activate project-env
python -m ipykernel install --user --name=project-env --display-name "Python (project-env)"
```

Then when you run `results.ipynb` be sure to select `Python (project-env)` as your kernel

Running the Jupyter notebook with an altered filename will rerun the data pull and modeling process
- NOTE: the data pull errs on the side of caution and inserts a lot of sleep to not disturb the OpenLibrary API, making the data pull take ~1.5 hrs for 300 sample per genre

Running the jupyter notebook with the current filename will use data from the pkl files in `src/data/` if you wish to run the modeling without using the pre modeled data, you may do so by renaming or removing the _modeled.pkl file and the script will find the raw data and perform the cluster model fit