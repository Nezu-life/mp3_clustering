import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

def standardize_data(df):
    print("Standardizing the data.")
    scaler = StandardScaler()

    return scaler.fit_transform(df)

def perform_pca(data):
    print("Running PCA analysis")

    pca = PCA(n_components=0.99)
    pca.fit(data)
    transformed_data = pca.transform(data)
    print(f"Number of components after PCA: {pca.n_components_}")
    
    return transformed_data

def elbow_method(data, min_k, max_k):
    print("KMeans in progress.")

    wcss = []
    for i in tqdm(range(min_k, max_k + 1), desc="Elbow Method Progress"):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(min_k, max_k + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    return wcss

def main():
    if len(sys.argv) != 4:
        print("Usage: python elbow_KMeans_mp3.py <csv_file> <min_k> <max_k>")
        sys.exit(1)

    # Explanation of features
    print("Welcome!")
    print("")
    print("This program will find the optimal number of cluster for your MP3 files.")
    print("1. It standardizes the variables, to make sure they are all in the same scale.")
    print("2. It runs a principal component analysis (PCA) of the features of each MP3 file.")
    print("3. It runs the KMeans clustering algorithm, varying the number of clusters from min_k to max_k.")
    print("4. You get a plot with the results to select the best k value for your data.")
    print("")
    print("By Nezu Life Sciences, November 2023")
    print("")

    csv_file = sys.argv[1]
    min_k = int(sys.argv[2])
    max_k = int(sys.argv[3])

    df = pd.read_csv(csv_file, usecols=lambda column: column not in ['filename'], quotechar="\"")
    
    standardized_data = standardize_data(df)
    
    pca_data = perform_pca(standardized_data)

    elbow_method(pca_data, min_k, max_k)

    print("Consider the 'elbow' in the plot to choose the best k value.")

if __name__ == "__main__":
    main()
