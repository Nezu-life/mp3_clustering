<img src="logo.png" alt="Logo" width="100" height="100">

# MP3 Clustering

First of all, thanks for your interest in using our programs.

This pipeline helps you cluster any number of MP3 files, according to their similarity.

First, it extracts several features from the individual MP3 files and creates a CSV with those features.

- MFCCs: Mel-Frequency Cepstral Coefficients.
- Chroma: Pertains to the 12 different pitch classes.
- Spectral Contrast: Difference in amplitude between peaks and valleys in the sound spectrum.
- Tempo: The speed or pace of the given piece.

Second, it runs the KMeans algorithm using different values of K. This helps identify the best value of K, using the so-called <a href="https://en.wikipedia.org/wiki/Elbow_method_(clustering)">elbow method</a>.

Third, after identifying the best number of clusters, you run the KMeans algorithm, and the last script will create one folder for each cluster, and will move the MP3 files inside each new folder.

## How to run it

1. Clone the repo
   ```sh
   git clone https://github.com/Nezu-life/mp3_clustering.git
   cd mp3_clustering
   ```
2. Run the MP3 featurizer in the directory of your MP3 files
   ```sh
   python3 mp3_featurizer.py <mp3_directory> <output_file.csv>
   ```
3. Run the KMeans with different K-values, to identify the best number of clusters.
   ```sh
   python3 ./src/elbow_KMeans_mp3.py <output_file.csv> <min_k> <max_k>
   ```

4. Run the KMeans with the best number of clusters (run it inside the directory with all MP3 files).
   ```sh
   python3 ./src/cluster_and_move_mp3_files.py <output_file.csv> <k>
   ```

NOTE: Single-quotes, double quotes or commas in the names of the MP3 files can throw the whole thing off-balance. So make sure the MP3 file names have no quotes or commas.

## Dependencies

- Pandas (2.1.1)
- Sklearn (1.3.1)
- Matplotlib (3.8.0)
- TQDM (4.66.1)
- Librosa (0.10.1)
- Numpy (1.23.5)

## Ready to go?

Comments, suggestions, forks and improvements are very much welcome.

Made with ❤️  by Tiago Lopes, PhD - the founder of Nezu Life Sciences.



<p align="right">(<a href="#readme-top">back to top</a>)</p>
