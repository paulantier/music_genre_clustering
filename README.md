# music_genre_clustering
Creating 2D maps of different music genres using CNN feature extraction on MEL-spectrograms of music tracks

The plan was to see how coherent music genre classification is in comparison of sound similarity.

The first step is to scrape a lot of labeled 30 seconds musical excerpts using the Spotify API, then to process the labels and reduce them to around 30 distinct genres.

After i turned the audio .wav files into images of their MEL-spectograms that are supposed to be a decent representation of the differents frequencies that humans distinguish.

With these pre-processing steps done, a basic classification CNN is trained to associate the images of the MEL-spectograms to their music genre. A few data augmentations were tried.

The final goal is to put a new spectogram into the CNN backbone and to get the values of the last featuremap.
Gathering those values for a lot of differents music excerpts allows us to get a multi-dimensional representation of the audio similarities and differences of differents tracks.
Using PCA (principal component analysis) on this data can help us reduce the data space dimensions to 2 or 3 for visualization purposes without losing too much information of similarity and difference.
Linking every data point in this new space to a color related to their music genre helps visualizing how similar music genres are, regarding sound similarity

