from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        melahirkan = float(request.form['melahirkan'])
        glukosa = float(request.form['glukosa'])
        darah = float(request.form['darah'])
        kulit = float(request.form['kulit'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        riwayat = float(request.form['riwayat'])
        umur = float(request.form['umur'])

        # Masukkan data ke dalam array dan normalisasi dengan scaler
        datas = np.array([[melahirkan, glukosa, darah, kulit, insulin, bmi, riwayat, umur]])
        datas = scaler.transform(datas)  # Normalisasi

        # Prediksi menggunakan model KNN
        isDiabetes = model.predict(datas)

        return render_template('hasil.html', finalData=isDiabetes)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
