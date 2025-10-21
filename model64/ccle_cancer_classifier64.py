"""
CCLE Kanser Hücre Hattı Sınıflandırma Projesi
=============================================
Bu proje, CCLE (Cancer Cell Line Encyclopedia) verilerini kullanarak kanser hücre hatlarını
türlerine göre sınıflandıran bir makine öğrenmesi modeli oluşturur. Kod aşağıdaki adımları içerir:
1. CCLE veri setinin indirilmesi
2. Veri yükleme ve ön işleme
3. Veri analizi ve görselleştirme
4. Özellik seçimi ve boyut indirgeme
5. Farklı makine öğrenmesi modellerinin eğitimi ve değerlendirilmesi
6. En iyi modelin kaydedilmesi ve test edilmesi
7. Sonuçların görselleştirilmesi
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import joblib
from imblearn.over_sampling import SMOTE
import requests
import shutil
import tarfile
import gzip
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

class CCLECancerClassifier:
    def __init__(self, data_dir="./ccle_data"):
        """
        CCLE Kanser Hücre Hatti Siniflandirma modelini başlatir
        
        Args:
            data_dir: CCLE verilerinin bulunduğu/indirileceği dizin
        """
        self.data_dir = data_dir
        self.expression_file = None
        self.annotation_file = None
        self.exp_data = None
        self.annotation_data = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = None
        self.selected_features = None
        self.cancer_types = None
        
        # Veri dizini oluştur
        os.makedirs(data_dir, exist_ok=True)
    

    def load_data(self):
        """
        CCLE gen ekspresyon ve hücre hatti bilgilerini yükler
        """
        if not os.path.exists(self.expression_file) or not os.path.exists(self.annotation_file):
            print("CCLE verileri bulunamadi.")
            return False
        
        print("Veriler yukleniyor...")
        
        try:
            # Gen ekspresyon verilerini yükle
            self.exp_data = pd.read_csv(self.expression_file, sep='\t', index_col=0, encoding='latin1')
            print(f"Gen ekspresyon verisi yuklendi. Boyut: {self.exp_data.shape}")
            
            # Hücre hattı bilgilerini yükle
            self.annotation_data = pd.read_csv(self.annotation_file, sep='\t', encoding='latin1')
            print(f"Hucre hatti bilgileri yuklendi. Boyut: {self.annotation_data.shape}")
            
            # NaN değerleri kontrol et
            if self.annotation_data['CCLE_ID'].isna().any():
                print(f"UYARI: {self.annotation_data['CCLE_ID'].isna().sum()} adet eksik CCLE_ID bulundu. Bunlar kaldiriliyor.")
                self.annotation_data = self.annotation_data.dropna(subset=['CCLE_ID'])
            
            if self.annotation_data['Name'].isna().any():
                print(f"UYARI: {self.annotation_data['Name'].isna().sum()} adet eksik Name bulundu. Bunlar kaldiriliyor.")
                self.annotation_data = self.annotation_data.dropna(subset=['Name'])
        
            # CCLE_ID ve hücre isimlerini alın
            cell_ids = self.annotation_data['CCLE_ID'].tolist()
            cell_names = self.annotation_data['Name'].tolist()
            
            # Stringlerden emin olmak için dönüştürme yapın
            cell_ids = [str(x) for x in cell_ids]
            cell_names = [str(x) for x in cell_names]
            
            print("Ilk birkac ekspresyon sutunu:", list(self.exp_data.columns)[:5])
            print("Ilk birkac CCLE_ID:", cell_ids[:5])
            
            # Sadece hücre hattı sütunlarını alın, diğer sütunları kaldırın
            if 'transcript_ids' in self.exp_data.columns:
                # Transcript ID'leri ayrı bir değişkende saklayın
                transcript_ids = self.exp_data['transcript_ids']
                self.exp_data = self.exp_data.drop(columns=['transcript_ids'])
            
            # Ekspresyon verisini transpoze et (hücre hatları satırlar, genler sütunlar olacak)
            self.exp_data = self.exp_data.transpose()
            print(f"Transpoze sonrasi ekspresyon verisi boyutu: {self.exp_data.shape}")
            
            # Şimdi verileri eşleştirmeye çalışalım
            # Ekspresyon veri indekslerini kontrol et
            exp_indices = list(self.exp_data.index)
            print(f"Ilk birkac ekspresyon indeksi: {exp_indices[:5]}")
            
            # Annotation veri indekslerini CCLE_ID olarak ayarla
            self.annotation_data.set_index('CCLE_ID', inplace=True)
            anno_indices = list(self.annotation_data.index)
            print(f"Ilk birkac annotation indeksi: {anno_indices[:5]}")
            
            # Ekspresyon indekslerini "_" işaretine göre ayır ve ilk kısmı al
            # Örneğin "22RV1_PROSTATE" -> "22RV1"
            exp_cell_names = []
            for idx in exp_indices:
                if '_' in str(idx):
                    exp_cell_names.append(str(idx).split('_')[0])
                else:
                    exp_cell_names.append(str(idx))
            
            # Ekspresyon indeksleri ile CCLE_ID'ler arasındaki ilişkiyi bulmaya çalış
            # İsim içerme kontrolü yap
            matches = []
            exp_to_ccle_map = {}
            
            for i, exp_name in enumerate(exp_cell_names):
                for j, (ccle_id, cell_name) in enumerate(zip(cell_ids, cell_names)):
                    try:
                        # String'e dönüştürme ve NaN kontrolü
                        exp_name_str = str(exp_name).lower()
                        cell_name_str = str(cell_name).lower()
                        
                        # Boş string kontrolü
                        if not exp_name_str or not cell_name_str or exp_name_str == 'nan' or cell_name_str == 'nan':
                            continue
                        
                        # İçerme kontrolü
                        if exp_name_str in cell_name_str or cell_name_str in exp_name_str:
                            matches.append((exp_indices[i], ccle_id, exp_name, cell_name))
                            exp_to_ccle_map[exp_indices[i]] = ccle_id
                            break
                    except Exception as e:
                        print(f"Eşleştirme hatasi: {e} - exp_name: {exp_name}, cell_name: {cell_name}")
                        continue
            
            print(f"Toplam {len(matches)} eslesme bulundu.")
            
            if len(matches) == 0:
                print("Eslesme bulunamadi. Verileri manuel olarak incelemek gerekebilir.")
                
                # Alternatif eşleştirme yöntemi: CCLE_ID'leri doğrudan kullanmayı deneyelim
                print("Alternatif eşleştirme deneniyor: Dorudan CCLE_ID'leri kontrol ediyoruz.")
                
                # Ekspresyon indekslerinin bazıları doğrudan CCLE_ID'lere karşılık gelebilir
                direct_matches = set(exp_indices) & set(anno_indices)
                
                if direct_matches:
                    print(f"Dogrudan {len(direct_matches)} eslesme bulundu.")
                    
                    # Sadece doğrudan eşleşen verileri kullanın
                    self.exp_data = self.exp_data.loc[list(direct_matches)]
                    self.annotation_data = self.annotation_data.loc[list(direct_matches)]
                else:
                    print("Dorudan eşleştirme de başarisiz oldu.")
                    return False
            else:
                # Sadece eşleşen hücre hatlarını al
                match_exp_indices = [m[0] for m in matches]
                match_anno_indices = [m[1] for m in matches]
                
                # Filtrele
                self.exp_data = self.exp_data.loc[match_exp_indices]
                
                # Annotation verisi indeksinde olmayan değerleri temizle
                valid_anno_indices = [idx for idx in match_anno_indices if idx in self.annotation_data.index]
                
                if len(valid_anno_indices) < len(match_anno_indices):
                    print(f"UYARI: {len(match_anno_indices) - len(valid_anno_indices)} eslesme annotation verisinde bulunamadi.")
                
                if not valid_anno_indices:
                    print("Gecerli eslesme kalmadi!")
                    return False
                
                self.annotation_data = self.annotation_data.loc[valid_anno_indices]
                
                # Ekspresyon veri indekslerini CCLE_ID'ler ile değiştir
                new_indices = []
                for idx in match_exp_indices:
                    if idx in exp_to_ccle_map and exp_to_ccle_map[idx] in self.annotation_data.index:
                        new_indices.append(exp_to_ccle_map[idx])
                    else:
                        new_indices.append(idx)  # Eşleşme bulunamadıysa orijinal indeksi koru
                
                self.exp_data.index = new_indices
            
            # Şimdi eşleşme kontrolü yap
            common_ids = set(self.exp_data.index) & set(self.annotation_data.index)
            print(f"Eslesme sonrasi ortak ID sayisi: {len(common_ids)}")
            
            if len(common_ids) == 0:
                print("Eslestirme sonrasi bile ortak ID bulunamadi. Verileri manuel kontrol edin.")
                return False
            
            # Sadece ortak ID'leri içeren verileri al
            self.exp_data = self.exp_data.loc[list(common_ids)]
            self.annotation_data = self.annotation_data.loc[list(common_ids)]
            
            # Kanser türü etiketlerini belirle
            possible_label_columns = ["Site_Primary", "Pathology", "Histology", "Disease"]
            label_column = None
            
            for column in possible_label_columns:
                if column in self.annotation_data.columns:
                    label_column = column
                    break
            
            if label_column is None:
                print(f"UYARI: Kanser turu icin uygun sutun bulunamadi.")
                print("Mevcut sutunlar:", self.annotation_data.columns.tolist())
                return False
            
            print(f"Kanser turu sutunu olarak '{label_column}' kullanisiyor.")
            
            self.labels = self.annotation_data[label_column]
            self.cancer_types = self.labels.unique()
            print(f"Toplam {len(self.cancer_types)} farkli kanser turu bulundu.")
            
            # Çok az örneği olan kanser türlerini filtrele
            min_samples = 5
            type_counts = self.labels.value_counts()
            rare_types = type_counts[type_counts < min_samples].index.tolist()
            
            if rare_types:
                print(f"Asaidaki kanser turleri {min_samples}'ten az ornege sahip oldugu icin filtreleniyor.")
                valid_indices = ~self.labels.isin(rare_types)
                self.exp_data = self.exp_data[valid_indices]
                self.annotation_data = self.annotation_data[valid_indices]
                self.labels = self.labels[valid_indices]
                self.cancer_types = self.labels.unique()
                print(f"Filtreleme sonrasi {len(self.cancer_types)} kanser turu kaldi.")
            
            # Eksik değerleri doldur
            self.exp_data = self.exp_data.fillna(0)
            
            # Etiketleri kodla
            self.labels = pd.Series(self.label_encoder.fit_transform(self.labels), index=self.labels.index)
            
            return True
        
        except Exception as e:
            print(f"Veri yukleme hatasi: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_data(self):
        """
        Veri seti hakkında özet istatistikler ve görselleştirmeler
        """
        if self.exp_data is None or self.annotation_data is None:
            print("Veri henuz yuklenmedi. Once load_data() metodunu cagirin.")
            return
        
        print("\n=== Veri Analizi ===")
        
        # Kanser türlerinin dağılımı - düzeltilmiş kod
        label_column = None
        for col in ["Site_Primary", "Pathology", "Histology", "Disease"]:
            if col in self.annotation_data.columns:
                label_column = col
                break
        
        if label_column:
            cancer_counts = self.annotation_data[label_column].value_counts()
            print("\nKanser turlerinin dagilimi:")
            print(cancer_counts)
            
            plt.figure(figsize=(14, 8))
            ax = cancer_counts.plot(kind='bar')
            plt.title('CCLE Kanser Hucre Hatlarinin Turlere Gore Dagilimi')
            plt.ylabel('Hucre Hatti Sayisi')
            plt.xlabel('Kanser Turu')
            plt.xticks(rotation=45, ha='right')
            
            # Etiketleri görünür hale getir
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'cancer_distribution.png'))
            plt.close()
        else:
            print("Kanser turlerinin dagilimini gostermek icin uygun bir sutun bulunamadi.")
        
        # Ekspresyon verisinin dağılımı
        plt.figure(figsize=(10, 6))
        sns.histplot(self.exp_data.values.flatten(), bins=50, kde=True)
        plt.title('Gen Ekspresyon Deerlerinin Dagilimi')
        plt.xlabel('Ekspresyon Deeri')
        plt.ylabel('Frekans')
        plt.savefig(os.path.join(self.data_dir, 'expression_distribution.png'))
        plt.close()
        
        # Eksik değer analizi
        missing_values = self.exp_data.isnull().sum().sum()
        print(f"\nEksik deer sayisi: {missing_values}")
        
        # En çok değişen genler
        gene_variance = self.exp_data.var().sort_values(ascending=False)
        top_variable_genes = gene_variance.head(20)
        
        plt.figure(figsize=(12, 6))
        top_variable_genes.plot(kind='bar')
        plt.title('En Yuksek Varyansa Sahip 20 Gen')
        plt.ylabel('Varyans')
        plt.xlabel('Gen')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'top_variable_genes.png'))
        plt.close()
        
        print("\nEn yuksek varyansa sahip 10 gen:")
        print(top_variable_genes.head(10))
        
        # PCA analizi ile verileri 2D düzlemde görselleştirme
        try:
            # Çok sayıda gen olduğu için, hesaplama yükünü azaltmak için ilk 1000 geni kullanabiliriz
            sample_genes = self.exp_data.columns[:1000]
            sample_data = self.exp_data[sample_genes]
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(StandardScaler().fit_transform(sample_data))
            
            plt.figure(figsize=(12, 10))
            
            # Etiketlerin benzersiz değerlerini alın
            unique_labels = set(self.labels)
            
            # Her bir etiket için farklı bir renk kullanın
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            
            # Etiketleri sayısal indekslere dönüştür
            label_to_index = {label: i for i, label in enumerate(unique_labels)}
            
            for label in unique_labels:
                # Bu etikete sahip tüm örneklerin indekslerini bulun
                indices = self.labels[self.labels == label].index
                
                # Orijinal verideki indekslere karşılık gelen PCA indekslerini bulun
                pca_indices = [i for i, idx in enumerate(self.exp_data.index) if idx in indices]
                
                if pca_indices:  # Boş liste kontrolü
                    # PCA sonuçlarından ilgili noktalara ait verileri alın
                    x_values = [pca_result[i, 0] for i in pca_indices]
                    y_values = [pca_result[i, 1] for i in pca_indices]
                    
                    # Bu etiket için nokta grafiği çizin
                    label_name = self.label_encoder.inverse_transform([label])[0]
                    plt.scatter(x_values, y_values, 
                            label=label_name,
                            c=[cmap(label_to_index[label])], 
                            alpha=0.7, 
                            s=50)
            
            plt.title('PCA ile CCLE Hucre Hatlarinin Gorselleştirilmesi')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'pca_visualization.png'))
            plt.close()
            
        except Exception as e:
            print(f"PCA gorselleştirme hatasi: {str(e)}")
            # Detaylı hata mesajını yazdır
            import traceback
            traceback.print_exc()

    def preprocess_data(self, n_features=1000):
        """
        Verileri ön işleme: ölçeklendirme, özellik seçimi
        
        Args:
            n_features: Seçilecek özellik (gen) sayısı
        """
        if self.exp_data is None:
            print("Veri henuz yuklenmedi. Once load_data() metodunu cagirin.")
            return None, None, None, None
        
        print("\n=== Veri On Isleme ===")
        
        # ÖNEMLİ: Özellik seçimini ölçeklendirmeden ÖNCE yap 
        # En bilgilendirici genleri seç
        selector = SelectKBest(f_classif, k=min(n_features, self.exp_data.shape[1]))
        exp_data_selected = selector.fit_transform(self.exp_data, self.labels)
        
        # Seçilen genlerin indekslerini al
        selected_indices = selector.get_support(indices=True)
        self.selected_features = self.exp_data.columns[selected_indices]
        
        # Seçilen genleri içeren yeni bir DataFrame oluştur
        exp_data_selected_df = pd.DataFrame(
            exp_data_selected, 
            index=self.exp_data.index, 
            columns=self.selected_features
        )
        
        # ŞİMDİ sadece seçilen genler için ölçeklendirme yap
        exp_data_scaled = self.scaler.fit_transform(exp_data_selected_df)
        exp_data_scaled_df = pd.DataFrame(
            exp_data_scaled, 
            index=self.exp_data.index, 
            columns=self.selected_features
        )
        
        print(f"Ozellik secimi tamamlandi. {len(self.selected_features)} gen secildi.")
        
        # Eğitim ve test kümelerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            exp_data_scaled_df, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )
        
        print(f"Egitim seti boyutu: {X_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train, balance_data=True):
        """
        Farklı makine oğrenmesi modellerini eğit ve değerlendir
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            balance_data: Veri dengesizliği ile başa çıkmak için SMOTE uygula
            
        Returns:
            En iyi model ve skorlar
        """
        print("\n=== Model Egitimi ===")
        
        if balance_data:
            try:
                # Sınıf başına en az örnek sayısını hesaplayalım
                class_counts = pd.Series(y_train).value_counts()
                min_class_count = class_counts.min()
                
                print(f"Sinif basina en az ornek sayisi: {min_class_count}")
                
                # SMOTE için k_neighbors parametresini ayarlayalım
                # k_neighbors, sınıf başına örnek sayısından küçük olmalı
                k_neighbors = min(min_class_count - 1, 5)  # En az 1, en fazla 5
                
                if k_neighbors < 1:
                    print("UYARI: Bazi siniflarda cok az ornek var. SMOTE atlama.")
                    X_train_balanced, y_train_balanced = X_train, y_train
                else:
                    print(f"SMOTE uygulaniyor, k_neighbors={k_neighbors}...")
                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
                    print(f"SMOTE uygulandi. Yeni egitim seti boyutu: {X_train_balanced.shape}")
            except Exception as e:
                print(f"SMOTE hatasi: {str(e)}")
                print("SMOTE atlama, orijinal veri kullaniliyor.")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # Çapraz doğrulama için k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Modeller ve hiperparametreler
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'kernel': ['rbf', 'linear']
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=500),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']
                }
            }
        }
        
        best_score = 0
        best_model = None
        results = {}
        
        for name, config in models.items():
            print(f"\nEgitiliyor: {name}")
            try:
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv, 
                    scoring='accuracy', 
                    n_jobs=-1, 
                    verbose=1
                )
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                best_params = grid_search.best_params_
                best_score_model = grid_search.best_score_
                
                results[name] = {
                    'best_params': best_params,
                    'best_score': best_score_model
                }
                
                print(f"{name} - En iyi parametreler: {best_params}")
                print(f"{name} - Capraz dogrulama skoru: {best_score_model:.4f}")
                
                if best_score_model > best_score:
                    best_score = best_score_model
                    best_model = grid_search.best_estimator_
                    best_model_name = name
            except Exception as e:
                print(f"{name} egitimi sirasinda hata: {str(e)}")
                continue
        
        if best_model is None:
            print("Hicbir model başariyla egitilmedi. Varsayilan model olarak RandomForest kullaniliyor.")
            best_model = RandomForestClassifier(random_state=42).fit(X_train_balanced, y_train_balanced)
            best_model_name = "RandomForest (default)"
        else:
            print(f"\nEn iyi model: {best_model_name} (Skor: {best_score:.4f})")
        
        self.model = best_model
        
        return best_model, results

    def evaluate_model(self, model, X_test, y_test):
        """
        Modeli test seti üzerinde değerlendir
        
        Args:
            model: Eğitilmiş model
            X_test: Test özellikleri
            y_test: Test etiketleri
            
        Returns:
            Tahminler ve doğruluk skoru
        """
        print("\n=== Model Degerlendirme ===")
        
        # Tahminleri yap
        y_pred = model.predict(X_test)
        
        # Skorları hesapla
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test dogrulugu: {accuracy:.4f}")
        
        # Sınıflandırma raporu
        print("\nSiniflandirma Raporu:")
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.cancer_types))]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Karmaşıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Tahmin Edilen Etiket')
        plt.ylabel('Gercek Etiket')
        plt.title('Karmasiklik Matrisi')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Yanlış sınıflandırılan örnekleri incele
        misclassified_indices = np.where(y_test != y_pred)[0]
        misclassified_data = {
            'Sample ID': y_test.index[misclassified_indices],
            'True Cancer Type': self.label_encoder.inverse_transform(y_test.iloc[misclassified_indices]),
            'Predicted Cancer Type': self.label_encoder.inverse_transform(y_pred[misclassified_indices])
        }
        misclassified_df = pd.DataFrame(misclassified_data)
        
        # Yanlış sınıflandırılan örnekleri kaydet
        misclassified_file = os.path.join(self.data_dir, 'misclassified_samples.csv')
        misclassified_df.to_csv(misclassified_file, index=False)
        print(f"\nYanlis siniflandirilan ornekler kaydedildi: {misclassified_file}")
        
        return y_pred, accuracy

    def visualize_embeddings(self, X, y, method='tsne'):
        """
        Yüksek boyutlu verileri 2D düzlemde görselleştirir
        
        Args:
            X: Özellikler
            y: Etiketler
            method: Boyut indirgeme metodu ('tsne' veya 'pca')
        """
        print(f"\n=== Veri Gorsellestirme ({method}) ===")
        
        if method == 'tsne':
            # t-SNE ile boyut indirgeme
            embedding = TSNE(n_components=2, random_state=42)
            X_embedded = embedding.fit_transform(X)
            title = 't-SNE ile Kanser Turlerinin Dagilimi'
        else:
            # PCA ile boyut indirgeme
            embedding = PCA(n_components=2)
            X_embedded = embedding.fit_transform(X)
            title = 'PCA ile Kanser Turlerinin Dagilimi'
        
        # Etiketleri numpy array'e dönüştür
        y_array = np.array(y)
        
        # Görselleştirme
        plt.figure(figsize=(14, 10))
        cmap = plt.cm.get_cmap('tab20', len(self.cancer_types))
        
        for i, cancer_type in enumerate(np.unique(y_array)):
            indices = np.where(y_array == cancer_type)[0]
            plt.scatter(X_embedded[indices, 0], X_embedded[indices, 1], 
                       label=self.label_encoder.inverse_transform([cancer_type])[0],
                       c=[cmap(i)], alpha=0.7, s=50)
        
        plt.title(title)
        plt.xlabel('Bilesen 1')
        plt.ylabel('Bilesen 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, f'{method}_visualization.png'))
        plt.close()

    def save_model(self, filename='ccle_cancer_classifier_model.pkl'):
        """
        Eğitilmiş modeli kaydet
        """
        if self.model is None:
            print("Model henuz egitilmedi.")
            return
        
        model_path = os.path.join(self.data_dir, filename)
        
        # Modeli ve gerekli bileşenleri kaydet
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,  # Seçilen gen listesi
            'cancer_types': self.cancer_types
        }
        
        joblib.dump(model_package, model_path)
        print(f"Model basariyla kaydedildi: {model_path}")

    def load_model(self, filename='ccle_cancer_classifier_model.pkl'):
        """
        Kaydedilmiş modeli yükle
        
        Args:
            filename: Model dosyasının adı
        """
        model_path = os.path.join(self.data_dir, filename)
        
        try:
            model_package = joblib.load(model_path)
            
            self.model = model_package['model']
            self.scaler = model_package['scaler']
            self.label_encoder = model_package['label_encoder']
            self.selected_features = model_package['selected_features']
            self.cancer_types = model_package['cancer_types']
            
            print(f"Model basariyla yuklendi: {model_path}")
            return True
        
        except Exception as e:
            print(f"Model yukleme hatasi: {str(e)}")
            return False

    def predict_sample(self, sample_data):
        """
        Yeni bir örneğin kanser türünü tahmin et
        
        Args:
            sample_data: Yeni örneğin gen ekspresyon verisi (DataFrame)
            
        Returns:
            Tahmin edilen kanser türü ve olasılık
        """
        if self.model is None:
            print("Model henuz egitilmedi veya yuklenmedi.")
            return None, None
        
        # Ortak genleri kontrol et
        common_genes = set(sample_data.columns) & set(self.selected_features)
        
        if len(common_genes) < len(self.selected_features):
            print(f"UYARI: Ornekte {len(self.selected_features) - len(common_genes)} gen eksik.")
            
            # Eksik genleri sıfır değeriyle doldur
            for gene in set(self.selected_features) - common_genes:
                sample_data[gene] = 0
        
        # Sadece seçilen genleri al
        sample_data = sample_data[self.selected_features]
        
        # Ölçeklendir
        sample_scaled = self.scaler.transform(sample_data)
        
        # Tahmin yap
        prediction = self.model.predict(sample_scaled)[0]
        probabilities = self.model.predict_proba(sample_scaled)[0]
        
        # Etiket kodunu gerçek kanser türüne çevir
        cancer_type = self.label_encoder.inverse_transform([prediction])[0]
        probability = max(probabilities)
        
        return cancer_type, probability
        
    def run_pipeline(self, n_features=1000, balance_data=True, download=True):
        """
        Tüm işlem hattını çalıştır: veri indirme, yükleme, ön işleme, model egitimi, degerlendirme ve kaydetme
        
        Args:
            n_features: Seçilecek gen sayısı
            balance_data: Veri dengesizliğini düzeltmek için SMOTE uygulansın mı
            download: Veri indirilsin mi
            
        Returns:
            Eğitilmiş model ve doğruluk skoru
        """
        # Verileri indir (isteğe bağlı)
        if download:
            success = self.download_ccle_data()
            if not success:
                print("Veri indirme basarisiz. Islem durduruluyor.")
                return None, 0
        
        # Verileri yükle
        success = self.load_data()
        if not success:
            print("Veri yukleme basarisiz. Islem durduruluyor.")
            return None, 0
        
        # Verileri analiz et
        self.analyze_data()
        
        # Verileri ön işle
        X_train, X_test, y_train, y_test = self.preprocess_data(n_features=n_features)
        
        # Verileri görselleştir
        self.visualize_embeddings(X_train, y_train, method='tsne')
        self.visualize_embeddings(X_train, y_train, method='pca')
        
        # Modelleri eğit
        best_model, results = self.train_models(X_train, y_train, balance_data=balance_data)
        
        # Modeli değerlendir
        y_pred, accuracy = self.evaluate_model(best_model, X_test, y_test)
        
        # Modeli kaydet
        self.save_model()
        
        print("\n=== Islem Hatti Tamamlandi ===")
        print(f"Dogruluk: {accuracy:.4f}")
        
        return best_model, accuracy
    
    # Kullanım örnegi
if __name__ == "__main__":
    # Sınıflandırıcıyı başlat
    classifier = CCLECancerClassifier(data_dir="./ccle_data")

    # Dosya yollarını düzgün bir şekilde ayarla (eğik çizgi kullanarak)
    classifier.expression_file = "./ccle_data/CCLE_expression.csv"
    classifier.annotation_file = "./ccle_data/sample_info.csv"
    
    # Tüm işlem hattını çalıştır
    model, accuracy = classifier.run_pipeline(
        n_features=1000,     # Seçilecek gen sayısı
        balance_data=True,   # Veri dengesizliğini düzelt
        download=False       # CCLE verilerini indir
    )
    
    print("\n=== Proje Tamamlandi ===")
    print(f"Model kaydedildi. Test dogrulugu: {accuracy:.4f}")