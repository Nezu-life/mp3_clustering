import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shutil
from sklearn.decomposition import PCA


def standardize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def perform_pca(data):
    pca = PCA(n_components=0.99)
    pca.fit(data)
    transformed_data = pca.transform(data)
    print(f"Number of components after PCA: {pca.n_components_}")
    return transformed_data

def create_directories(num_clusters):
    for i in range(num_clusters):
        os.makedirs(f'cluster_{i}', exist_ok=True)

def move_files_to_clusters(filenames, labels):
    for filename, label in zip(filenames, labels):
        src = filename
        dst = f'cluster_{label}/{filename}'
        shutil.move(src, dst)

def main():
    if len(sys.argv) != 3:
        print("Usage: python cluster_and_move_mp3_files.py <csv_file> <k>")
        sys.exit(1)

    # Explanation of features
    print("Welcome!")
    print("")
    print("This program will cluster and move your MP3 files.")
    print("1. It standardizes the variables, to make sure they are all in the same scale.")
    print("2. It runs a principal component analysis (PCA) of the features of each MP3 file.")
    print("3. It runs the KMeans clustering algorithm with k clusters.")
    print("4. it creates k directories and moves the MP3 files to those folders.")
    print("Each file will be copied to the folder it belongs, depending on the cluster it was assigned.")
    print("Files in the same cluster directory are similar to each other.")
    print("")
    print("By Nezu Life Sciences, November 2023")
    print("")

    csv_file = sys.argv[1]
    k = int(sys.argv[2])

    df = pd.read_csv(csv_file)
    filenames = df.iloc[:, 0].tolist()
    data = df.drop(df.columns[0], axis=1)

    # Standardize the data
    standardized_data = standardize_data(data)

    # Perform PCA
    pca_data = perform_pca(standardized_data)

    # KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)
    labels = kmeans.labels_

    # Create directories and move files according to labels.
    create_directories(k)
    move_files_to_clusters(filenames, labels)

    print(f"Files have been moved to {k} clusters.")

if __name__ == "__main__":
    main()
