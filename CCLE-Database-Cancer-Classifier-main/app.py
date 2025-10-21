"""
CCLE Kanser Sınıflandırıcı Web Uygulaması
=======================================
Bu Flask uygulaması, eğitilmiş CCLE kanser sınıflandırıcısını web arayüzü ile kullanmayı sağlar.
Uygulama aşağıdaki özellikleri sunar:
1. Eğitilmiş modeli kullanarak yeni gen ekspresyon verileri için kanser türü tahmini
2. Gen ekspresyon dosyası yükleme ve toplu tahmin yapma
3. Tahmin sonuçlarını görselleştirme
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')  # Render için
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json

app = Flask(__name__)
app.secret_key = 'ccle_classifier_secret_key'

# Uygulama ayarları
MODEL_DIR = "./ccle_data"  # Modelin bulunduğu dizin
MODEL_FILE = "ccle_cancer_classifier_model.pkl"  # Model dosyası
UPLOAD_FOLDER = "./uploads"  # Yüklenen dosyalar için klasör
ALLOWED_EXTENSIONS = {'csv', 'tsv', 'txt'}  # İzin verilen dosya uzantıları

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Gerekli klasörleri oluştur
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Yüklenen dosya uzantısını kontrol et
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Modeli yükle
def load_model():
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    try:
        model_package = joblib.load(model_path)
        return model_package
    except Exception as e:
        print(f"Model yükleme hatası: {str(e)}")
        return None

# Tahmin yap
def predict_cancer_type(sample_data, model_package):
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoder = model_package['label_encoder']
    selected_features = model_package['selected_features']
    
    # Ortak genleri kontrol et
    common_genes = set(sample_data.columns) & set(selected_features)
    
    # Eksik genleri doldur
    for gene in set(selected_features) - common_genes:
        sample_data[gene] = 0
    
    # Sadece seçilen genleri al
    sample_data = sample_data[selected_features]
    
    # Ölçeklendir
    sample_scaled = scaler.transform(sample_data)
    
    # Tahmin yap
    predictions = model.predict(sample_scaled)
    probabilities = model.predict_proba(sample_scaled)
    
    # Etiket kodlarını gerçek kanser türlerine çevir
    cancer_types = label_encoder.inverse_transform(predictions)
    
    results = []
    for i, (cancer_type, prob_dist) in enumerate(zip(cancer_types, probabilities)):
        max_prob = max(prob_dist)
        max_prob_index = np.argmax(prob_dist)
        
        # Her bir sınıf için olasılıkları al
        class_probabilities = {}
        for j, prob in enumerate(prob_dist):
            class_name = label_encoder.inverse_transform([j])[0]
            class_probabilities[class_name] = float(prob)
        
        result = {
            'sample_id': sample_data.index[i] if i < len(sample_data.index) else f"Sample_{i}",
            'predicted_cancer': cancer_type,
            'confidence': float(max_prob),
            'class_probabilities': class_probabilities
        }
        results.append(result)
    
    return results

# Görselleştirme için fonksiyon
def create_prediction_chart(results):
    # Sonuçları DataFrame'e dönüştür
    df = pd.DataFrame([
        {'Sample': r['sample_id'], 'Cancer Type': r['predicted_cancer'], 'Confidence': r['confidence']}
        for r in results
    ])
    
    # En yüksek olasılıklı kanser türüne göre sırala
    df = df.sort_values('Confidence', ascending=False)
    
    # Görselleştirme
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Confidence', y='Sample', hue='Cancer Type', data=df)
    plt.title('Kanser Türü Tahmin Sonuçları')
    plt.xlabel('Güven Skoru')
    plt.tight_layout()
    
    # Grafiği base64 olarak kodla
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()
    
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Örnek veri tahmini
@app.route('/predict', methods=['POST'])
def predict():
    model_package = load_model()
    
    if model_package is None:
        return jsonify({
            'error': 'Model yüklenemedi. Lütfen model dosyasının doğru konumda olduğunu kontrol edin.'
        }), 500
    
    try:
        # JSON verisi al
        data = request.get_json()
        
        # Gen ekspresyon verisini DataFrame'e dönüştür
        sample_data = pd.DataFrame(data['expression'])
        
        # Tahmin yap
        results = predict_cancer_type(sample_data, model_package)
        
        return jsonify({
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Tahmin hatası: {str(e)}'
        }), 500

# Dosya yükleme ve tahmin
@app.route('/upload', methods=['POST'])
def upload_file():
    model_package = load_model()
    
    if model_package is None:
        flash('Model yüklenemedi. Lütfen model dosyasının doğru konumda olduğunu kontrol edin.', 'error')
        return redirect(url_for('index'))
    
    # Dosya yüklendi mi kontrol et
    if 'file' not in request.files:
        flash('Dosya bulunamadı', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Dosya adı boş mu kontrol et
    if file.filename == '':
        flash('Dosya seçilmedi', 'error')
        return redirect(url_for('index'))
    
    # Geçerli bir dosya mı kontrol et
    if file and allowed_file(file.filename):
        # Dosyayı yükle
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Dosya uzantısına göre ayırıcıyı belirle
            separator = '\t' if file.filename.endswith('.tsv') else ','
            
            # Dosyayı oku
            sample_data = pd.read_csv(file_path, sep=separator, index_col=0, encoding='utf-8')
            
            # Satır/sütun düzeni kontrolü
            # CCLE verilerinde satırlar hücre hatları, sütunlar genler olmalı
            # Eğer tersiyse, çevir
            if sample_data.shape[0] > sample_data.shape[1]:
                # Muhtemelen sütunlar hücre hatları, satırlar genler (transpose edilmeli)
                sample_data = sample_data.transpose()
                flash('Gen ve hücre hattı düzeni otomatik olarak algılandı ve düzeltildi.', 'info')
            
            # Tahmin yap
            results = predict_cancer_type(sample_data, model_package)
            
            # Grafiği oluştur
            chart_image = create_prediction_chart(results)
            
            return render_template('results.html', 
                                  results=results, 
                                  chart_image=chart_image,
                                  filename=file.filename)
        
        except Exception as e:
            flash(f'Dosya işleme hatası: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    flash('İzin verilmeyen dosya türü', 'error')
    return redirect(url_for('index'))

# API için dosya yükleme
@app.route('/api/upload', methods=['POST'])
def api_upload_file():
    model_package = load_model()
    
    if model_package is None:
        return jsonify({
            'error': 'Model yüklenemedi. Lütfen model dosyasının doğru konumda olduğunu kontrol edin.'
        }), 500
    
    # Dosya yüklendi mi kontrol et
    if 'file' not in request.files:
        return jsonify({
            'error': 'Dosya bulunamadı'
        }), 400
    
    file = request.files['file']
    
    # Dosya adı boş mu kontrol et
    if file.filename == '':
        return jsonify({
            'error': 'Dosya seçilmedi'
        }), 400
    
    # Geçerli bir dosya mı kontrol et
    if file and allowed_file(file.filename):
        # Dosyayı yükle
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            # Dosya uzantısına göre ayırıcıyı belirle
            separator = '\t' if file.filename.endswith('.tsv') else ','
            
            # Dosyayı oku
            sample_data = pd.read_csv(file_path, sep=separator, index_col=0, encoding='utf-8')
            
            # Satır/sütun düzeni kontrolü ve düzeltme
            if sample_data.shape[0] > sample_data.shape[1]:
                sample_data = sample_data.transpose()
            
            # Tahmin yap
            results = predict_cancer_type(sample_data, model_package)
            
            return jsonify({
                'results': results
            })
        
        except Exception as e:
            return jsonify({
                'error': f'Dosya işleme hatası: {str(e)}'
            }), 500
    
    return jsonify({
        'error': 'İzin verilmeyen dosya türü'
    }), 400

# Model bilgisi
@app.route('/model-info', methods=['GET'])
def model_info():
    model_package = load_model()
    
    if model_package is None:
        return jsonify({
            'error': 'Model yüklenemedi. Lütfen model dosyasının doğru konumda olduğunu kontrol edin.'
        }), 500
    
    # Model özelliklerini al
    cancer_types = [str(ct) for ct in model_package['cancer_types']]
    selected_features = model_package['selected_features'].tolist()
    
    return jsonify({
        'cancer_types': cancer_types,
        'feature_count': len(selected_features),
        'model_type': type(model_package['model']).__name__
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)