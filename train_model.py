import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Data latih contoh (dummy data)
X = np.array([
    [1, 85, 80, 20, 0, 25.5, 0.5, 30],
    [2, 120, 70, 35, 80, 30.0, 0.3, 40],
    [0, 90, 60, 25, 50, 28.0, 0.4, 35],
    [1, 100, 75, 30, 60, 27.5, 0.2, 45],
    [2, 130, 90, 40, 100, 32.0, 0.6, 50],
    [0, 95, 65, 22, 55, 26.5, 0.1, 25],
    [1, 110, 85, 28, 70, 29.0, 0.4, 38],
    [2, 140, 95, 45, 120, 35.0, 0.7, 55]
])

y = np.array([0, 1, 1, 0, 1, 0, 0, 1])  # 0: Tidak Diabetes, 1: Diabetes

# Split data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Buat model KNN
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Simpan model dan scaler ke file joblib
joblib.dump(knn_model, 'knn_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model KNN berhasil disimpan sebagai 'knn_model.joblib'")
print("Scaler berhasil disimpan sebagai 'scaler.joblib'")
