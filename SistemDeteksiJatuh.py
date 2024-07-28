# SistemDeteksiJatuh.py

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class SistemDeteksiJatuh:
    def __init__(self):
        self.X = None
        self.y = None
        self.X_scaled = None
        self.y_binary = None

    def baca_data(self, nama_file):
        print("Membaca data...")
        direktori_saat_ini = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(direktori_saat_ini, nama_file)
        df = pd.read_csv(path, header=None)
        
        X_df = df.drop([0, 1], axis=1)
        y_df = df.iloc[:, 1]
        
        self.X = X_df.to_numpy()
        self.y = y_df.to_numpy()
        print("Data berhasil dibaca.")

    def pra_proses_data(self):
        print("Melakukan pra-proses data...")
        scaler = MinMaxScaler()
        self.X_scaled = scaler.fit_transform(self.X)
        self.y_binary = np.where(self.y == 'F', 1, 0)
        print("Pra-proses data selesai.")

    def visualisasi_pca(self):
        print("Memvisualisasikan hasil PCA...")
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pcs[:, 0], pcs[:, 1], c=self.y_binary)
        plt.xlabel('Komponen Utama 1')
        plt.ylabel('Komponen Utama 2')
        plt.title('Visualisasi Data dengan PCA')
        plt.colorbar(label='Label (0: Tidak Jatuh, 1: Jatuh)')
        plt.show()

    def analisis_kmeans(self, maks_klaster=10):
        print("Melakukan analisis K-means...")
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(self.X_scaled)
        
        akurasi = []
        for n_klaster in range(2, maks_klaster + 1):
            kmeans = KMeans(n_clusters=n_klaster, random_state=42)
            y_kmeans = kmeans.fit_predict(pcs)
            akurasi.append(self.hitung_akurasi_mayoritas(self.y_binary, y_kmeans, n_klaster))
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, maks_klaster + 1), akurasi, marker='o')
        plt.xlabel('Jumlah Klaster')
        plt.ylabel('Akurasi')
        plt.title('Akurasi K-means vs Jumlah Klaster')
        plt.show()

    def hitung_akurasi_mayoritas(self, y_true, y_pred, n_clusters):
        akurasi = []
        for cluster in range(n_clusters):
            mask = (y_pred == cluster)
            if np.sum(mask) > 0:
                akurasi_cluster = np.mean(y_true[mask] == np.round(np.mean(y_true[mask])))
                akurasi.append(akurasi_cluster)
        return np.mean(akurasi)

    def klasifikasi_svm(self):
        print("Melakukan klasifikasi SVM...")
        pca = PCA(n_components=19)
        X_pca = pca.fit_transform(self.X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, self.y_binary, test_size=0.3, random_state=42)

        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        svm = SVC()
        grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Parameter terbaik SVM: {grid_search.best_params_}")
        print(f"Akurasi terbaik SVM: {grid_search.best_score_:.4f}")

        svm_terbaik = grid_search.best_estimator_
        akurasi_test = svm_terbaik.score(X_test, y_test)
        print(f"Akurasi SVM pada data uji: {akurasi_test:.4f}")

    def klasifikasi_mlp(self):
        print("Melakukan klasifikasi MLP...")
        pca = PCA(n_components=19)
        X_pca = pca.fit_transform(self.X_scaled)
        X_train, X_test, y_train, y_test = train_test_split(X_pca, self.y_binary, test_size=0.3, random_state=42)

        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }

        mlp = MLPClassifier(max_iter=1000)
        grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f"Parameter terbaik MLP: {grid_search.best_params_}")
        print(f"Akurasi terbaik MLP: {grid_search.best_score_:.4f}")

        mlp_terbaik = grid_search.best_estimator_
        akurasi_test = mlp_terbaik.score(X_test, y_test)
        print(f"Akurasi MLP pada data uji: {akurasi_test:.4f}")

def main():
    sistem = SistemDeteksiJatuh()
    sistem.baca_data("falldetection_dataset.csv")
    sistem.pra_proses_data()
    sistem.visualisasi_pca()
    sistem.analisis_kmeans()
    sistem.klasifikasi_svm()
    sistem.klasifikasi_mlp()

if __name__ == "__main__":
    main()
