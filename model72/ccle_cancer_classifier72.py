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
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
import requests
import shutil
import tarfile
import gzip
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

class CCLECancerClassifier:
    def __init__(self, data_dir="./ccle_data"):
        
        self.data_dir = data_dir
        self.expression_file = None
        self.annotation_file = None
        self.exp_data = None
        self.annotation_data = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()  # RobustScaler aykırı değerlere karşı daha dayanıklıdır
        self.model = None
        self.selected_features = None
        self.cancer_types = None
        self.feature_importance = None
        
        # Veri dizini oluştur
        os.makedirs(data_dir, exist_ok=True)
    

    def load_data(self):
        """
        CCLE gen ekspresyon ve hücre hattı bilgilerini yükler
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
            
            # Transcript ID'lerini ayır
            if 'transcript_ids' in self.exp_data.columns:
                transcript_ids = self.exp_data['transcript_ids']
                self.exp_data = self.exp_data.drop(columns=['transcript_ids'])
            
            # Ekspresyon verisini transpoze et
            self.exp_data = self.exp_data.transpose()
            print(f"Transpoze sonrasi ekspresyon verisi boyutu: {self.exp_data.shape}")
            
            # Ekspresyon veri indekslerini kontrol et
            exp_indices = list(self.exp_data.index)
            print(f"Ilk birkac ekspresyon indeksi: {exp_indices[:5]}")
            
            # Annotation veri indekslerini CCLE_ID olarak ayarla
            self.annotation_data.set_index('CCLE_ID', inplace=True)
            anno_indices = list(self.annotation_data.index)
            print(f"Ilk birkac annotation indeksi: {anno_indices[:5]}")
            
            # Ekspresyon indekslerinden hücre adlarını çıkar
            exp_cell_names = []
            for idx in exp_indices:
                if '_' in str(idx):
                    exp_cell_names.append(str(idx).split('_')[0])
                else:
                    exp_cell_names.append(str(idx))
            
            # Hücre hattı eşleştirmesi
            matches = []
            exp_to_ccle_map = {}
            
            for i, exp_name in enumerate(exp_cell_names):
                for j, (ccle_id, cell_name) in enumerate(zip(cell_ids, cell_names)):
                    try:
                        exp_name_str = str(exp_name).lower()
                        cell_name_str = str(cell_name).lower()
                        
                        if not exp_name_str or not cell_name_str or exp_name_str == 'nan' or cell_name_str == 'nan':
                            continue
                        
                        if exp_name_str in cell_name_str or cell_name_str in exp_name_str:
                            matches.append((exp_indices[i], ccle_id, exp_name, cell_name))
                            exp_to_ccle_map[exp_indices[i]] = ccle_id
                            break
                    except Exception as e:
                        continue
            
            print(f"Toplam {len(matches)} eslesme bulundu.")
            
            if len(matches) == 0:
                print("Eslesme bulunamadi. Alternatif eslestirme deneniyor...")
                
                # Doğrudan CCLE_ID eşleştirme
                direct_matches = set(exp_indices) & set(anno_indices)
                
                if direct_matches:
                    print(f"Dogrudan {len(direct_matches)} eslesme bulundu.")
                    self.exp_data = self.exp_data.loc[list(direct_matches)]
                    self.annotation_data = self.annotation_data.loc[list(direct_matches)]
                else:
                    print("Dogrudan eslestirme de basarisiz oldu.")
                    return False
            else:
                # Eşleşen hücre hatlarını al
                match_exp_indices = [m[0] for m in matches]
                match_anno_indices = [m[1] for m in matches]
                
                # Filtrele
                self.exp_data = self.exp_data.loc[match_exp_indices]
                
                # Annotation verisini filtrele
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
                        new_indices.append(idx)
                
                self.exp_data.index = new_indices
            
            # Eşleşme kontrolü
            common_ids = set(self.exp_data.index) & set(self.annotation_data.index)
            print(f"Eslesme sonrasi ortak ID sayisi: {len(common_ids)}")
            
            if len(common_ids) == 0:
                print("Eslestirme sonrasi ortak ID bulunamadi.")
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
            
            print(f"Kanser turu sutunu olarak '{label_column}' kullaniliyor.")
            
            self.labels = self.annotation_data[label_column]
            self.cancer_types = self.labels.unique()
            print(f"Toplam {len(self.cancer_types)} farkli kanser turu bulundu.")
            
            # Kanser türlerini sadeleştir (opsiyonel)
            # Burada daha jenerik gruplandırma yaparak sınıf sayısını azaltabilirsiniz
            # Örneğin: 'lung_adenocarcinoma' ve 'lung_squamous' -> 'lung'
            
            # Az örneği olan kanser türlerini filtrele
            min_samples = 8  # Minimum örnek sayısını arttırarak daha dengeli sınıflar elde edebiliriz
            type_counts = self.labels.value_counts()
            rare_types = type_counts[type_counts < min_samples].index.tolist()
            
            if rare_types:
                print(f"Asagidaki kanser turleri {min_samples}'ten az ornege sahip oldugu icin filtreleniyor.")
                valid_indices = ~self.labels.isin(rare_types)
                self.exp_data = self.exp_data[valid_indices]
                self.annotation_data = self.annotation_data[valid_indices]
                self.labels = self.labels[valid_indices]
                self.cancer_types = self.labels.unique()
                print(f"Filtreleme sonrasi {len(self.cancer_types)} kanser turu kaldi.")
            
            # Eksik değerleri doldur - median kullanarak
            self.exp_data = self.exp_data.fillna(self.exp_data.median())
            
            # Aykırı değerler için 3 sigma kuralı uygulayarak sınırlama
            for col in self.exp_data.columns:
                series = self.exp_data[col]
                mean = series.mean()
                std = series.std()
                self.exp_data[col] = series.clip(lower=mean - 3*std, upper=mean + 3*std)
            
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
        Veri seti hakkında ozet istatistikler ve gorselleştirmeler
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
        plt.title('Gen Ekspresyon Degerlerinin Dagilimi')
        plt.xlabel('Ekspresyon Degeri')
        plt.ylabel('Frekans')
        plt.savefig(os.path.join(self.data_dir, 'expression_distribution.png'))
        plt.close()
        
        # Eksik değer analizi
        missing_values = self.exp_data.isnull().sum().sum()
        print(f"\nEksik deger sayisi: {missing_values}")
        
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
        
        # PCA analizi - daha güvenli yaklaşım
        try:
            # Gen sayısı çok fazla olduğundan, en yüksek varyansa sahip 1000 geni kullanalım
            top_genes = gene_variance.index[:1000]
            sample_data = self.exp_data[top_genes]
            
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
            
            plt.title('PCA ile CCLE Hucre Hatlarinin Gorsellestirilmesi')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varyans)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varyans)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'pca_visualization.png'))
            plt.close()
            
            # Sınıflar arası korelasyon matrisi
            # Bu, hangi kanser türlerinin genetik olarak benzer olduğunu gösterebilir
            class_means = {}
            for label in unique_labels:
                indices = self.labels[self.labels == label].index
                class_data = self.exp_data.loc[indices]
                class_means[self.label_encoder.inverse_transform([label])[0]] = class_data.mean()
            
            # Sınıf ortalamalarını DataFrame'e dönüştür
            class_means_df = pd.DataFrame(class_means).T
            
            # Korelasyon matrisini hesapla
            corr_matrix = class_means_df.corr()
            
            # Korelasyon matrisini görselleştir
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Kanser Turleri Arasindaki Genetik Benzerlik (Korelasyon)')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'cancer_correlation.png'))
            plt.close()
            
        except Exception as e:
            print(f"PCA gorsellestirme hatasi: {str(e)}")

    def preprocess_data(self, n_features=500):
        """
        Verileri ön işleme: ölçeklendirme, özellik seçimi
        
        Args:
            n_features: Seçilecek özellik (gen) sayısı
        """
        if self.exp_data is None:
            print("Veri henuz yuklenmedi. Once load_data() metodunu cagirin.")
            return None, None, None, None
        
        print("\n=== Veri On Isleme ===")
        
        # Veriyi ölçeklendir - Robust Scaler
        exp_data_scaled = self.scaler.fit_transform(self.exp_data)
        exp_data_scaled = pd.DataFrame(exp_data_scaled, index=self.exp_data.index, columns=self.exp_data.columns)
        
        # Özellik seçimi - Geliştirilmiş yaklaşım
        print(f"Ozellik secimi yapiliyor...")
        
        # 1. İlk olarak varyansa göre ilk 2000 geni seç
        variance_selector = SelectKBest(lambda X, y: np.array(list(map(lambda x: np.var(x), X.T))), k=2000)
        exp_data_var_selected = variance_selector.fit_transform(exp_data_scaled, self.labels)
        var_selected_features = self.exp_data.columns[variance_selector.get_support()]
        exp_data_var_selected = pd.DataFrame(exp_data_var_selected, index=self.exp_data.index, 
                                           columns=var_selected_features)
        
        # 2. Ardından sınıf ayrımına göre seçim yap - hem ANOVA F-value hem de mutual information
        f_selector = SelectKBest(f_classif, k=n_features)
        mi_selector = SelectKBest(mutual_info_classif, k=n_features)
        
        exp_data_f_selected = f_selector.fit_transform(exp_data_var_selected, self.labels)
        exp_data_mi_selected = mi_selector.fit_transform(exp_data_var_selected, self.labels)
        
        f_selected_features = var_selected_features[f_selector.get_support()]
        mi_selected_features = var_selected_features[mi_selector.get_support()]
        
        # Her iki yöntemde de seçilen ortak genleri bul
        common_features = set(f_selected_features) & set(mi_selected_features)
        print(f"Her iki ozellik secim yonteminde de ortak olan gen sayisi: {len(common_features)}")
        
        # Ortak genler yeterli değilse, ikisinin birleşimini kullan
        if len(common_features) < n_features // 2:
            selected_features = list(set(f_selected_features) | set(mi_selected_features))[:n_features]
            print(f"Birlesim kullanilarak {len(selected_features)} gen secildi.")
        else:
            # Ortak genlere ek olarak, her iki yöntemden en iyi genleri ekle
            remaining_features = n_features - len(common_features)
            f_only = [f for f in f_selected_features if f not in common_features][:remaining_features//2]
            mi_only = [f for f in mi_selected_features if f not in common_features][:remaining_features//2]
            
            selected_features = list(common_features) + f_only + mi_only
            print(f"Ortak + ek genler kullanilarak {len(selected_features)} gen secildi.")
        
        # Seçilen genlere göre veriyi filtrele
        self.selected_features = selected_features
        exp_data_selected = exp_data_scaled[selected_features]
        
        print(f"Ozellik secimi tamamlandi. {len(self.selected_features)} gen secildi.")
        
        # Özellik önem derecelerini kaydet (RandomForest ile)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(exp_data_selected, self.labels)
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance
        
        # Top 20 en önemli genleri görselleştir
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('En Onemli 20 Gen (RandomForest Feature Importance)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'feature_importance.png'))
        plt.close()
        
        # Eğitim ve test kümelerine ayır - stratifiye örnekleme
        X_train, X_test, y_train, y_test = train_test_split(
            exp_data_selected, self.labels, test_size=0.25, random_state=42, stratify=self.labels
        )
        
        print(f"Egitim seti boyutu: {X_train.shape}")
        print(f"Test seti boyutu: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train, balance_data=True):
        """
        Farklı makine öğrenmesi modellerini eğit ve değerlendir
        
        Args:
            X_train: Eğitim özellikleri
            y_train: Eğitim etiketleri
            balance_data: Veri dengesizliği ile başa çıkmak için SMOTE uygula
            
        Returns:
            En iyi model ve skorlar
        """
        print("\n=== Model Eğitimi ===")
        
        # Sınıf dağılımını görselleştir
        class_distribution = Counter(y_train)
        print("Egitim seti sinif dagilimi:", class_distribution)
        
        # Veri dengeleme
        if balance_data:
            try:
                # Sınıf başına en az örnek sayısını hesaplayalım
                min_class_count = min(class_distribution.values())
                print(f"Sinif basina en az ornek sayisi: {min_class_count}")
                        
                # SMOTE için k_neighbors parametresini ayarlayalım
                k_neighbors = min(min_class_count - 1, 5)  # En az 1, en fazla 5
                
                if k_neighbors < 1:
                    print("UYARI: Bazi siniflarda cok az ornek var. SMOTE atlaniyor.")
                    X_train_balanced, y_train_balanced = X_train, y_train
                else:
                    print(f"SMOTE uygulaniyor, k_neighbors={k_neighbors}...")
                    
                    # SMOTE + ENN (Edited Nearest Neighbors) - Daha temiz sınır sağlar
                    smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors, random_state=42))
                    X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)
                    print(f"SMOTETomek uygulandi. Yeni egitim seti boyutu: {X_train_balanced.shape}")
                    print("Dengeli sinif dagilimi:", Counter(y_train_balanced))
            except Exception as e:
                print(f"Veri dengeleme hatasi: {str(e)}")
                print("Veri dengeleme atlaniyor, orijinal veri kullaniliyor.")
                X_train_balanced, y_train_balanced = X_train, y_train
        else:
           X_train_balanced, y_train_balanced = X_train, y_train
       
        # Çapraz doğrulama için k-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
       
        # Genişletilmiş model listesi
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': ['balanced', 'balanced_subsample', None]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1],
                    'kernel': ['rbf', 'linear'],
                    'class_weight': ['balanced', None]
                }
            },
            # Add these modifications to your NeuralNetwork parameters
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=42, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50, 25)],  # Smaller layers
                    'alpha': [0.001, 0.01, 0.1, 1.0],  # Stronger regularization
                    'learning_rate_init': [0.0001, 0.001, 0.01],  # Explicit learning rate
                    'activation': ['relu', 'tanh'],
                    'solver': ['adam'],  # Stick with adam
                    'batch_size': [32, 64, 128],  # Add batch size parameter
                    'early_stopping': [True],  # Enable early stopping
                    'n_iter_no_change': [10]  # Stop if no improvement for 10 iterations
                }
            }
        }
       
        best_score = 0
        best_model = None
        best_model_name = None
        results = {}
        
        # İlk aşama: Temel modelleri eğitme ve en iyilerini seçme
        for name, config in models.items():
            print(f"\nEgitiliyor: {name}")
            try:
                # GridSearchCV kullanarak hiperparametre optimizasyonu
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=cv, 
                    scoring='balanced_accuracy',  # Dengeli doğruluk skoru
                    n_jobs=-1, 
                    verbose=1
                )
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                best_params = grid_search.best_params_
                best_score_model = grid_search.best_score_
                
                results[name] = {
                    'best_params': best_params,
                    'best_score': best_score_model,
                    'model': grid_search.best_estimator_
                }
                
                print(f"{name} - En iyi parametreler: {best_params}")
                print(f"{name} - Capraz dogrulama skoru: {best_score_model:.4f}")
                
                # Daha detaylı metrikler için en iyi modeli test et
                scores = cross_val_score(grid_search.best_estimator_, X_train_balanced, y_train_balanced, 
                                        cv=cv, scoring='balanced_accuracy')
                print(f"{name} - Capraz dogrulama skorlari: {scores}")
                print(f"{name} - Ortalama: {scores.mean():.4f}, Standart sapma: {scores.std():.4f}")
                
                if best_score_model > best_score:
                    best_score = best_score_model
                    best_model = grid_search.best_estimator_
                    best_model_name = name
            except Exception as e:
                print(f"{name} egitimi sirasinda hata: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        if best_model is None:
            print("Hicbir model basariyla egitilmedi. Varsayilan model olarak RandomForest kullaniliyor.")
            best_model = RandomForestClassifier(random_state=42).fit(X_train_balanced, y_train_balanced)
            best_model_name = "RandomForest (default)"
        else:
            print(f"\nEn iyi tekli model: {best_model_name} (Skor: {best_score:.4f})")
        
        # İkinci aşama: Ensemble (Topluluk) modeli oluşturma - en iyi 3 modeli birleştir
        try:
            top_models = [(name, results[name]['model']) for name in 
                            sorted(results.keys(), key=lambda x: results[x]['best_score'], reverse=True)[:3]]
            
            if len(top_models) >= 2:  # En az 2 model varsa ensemble oluştur
                print("\nEn iyi 3 model kullanilarak Voting Classifier olusturuluyor...")
                
                # Tüm sınıflar için olasılık tahmini yapabilen modelleri kullan
                ensemble_models = [(name, model) for name, model in top_models 
                                    if hasattr(model, 'predict_proba')]
                
                if ensemble_models:
                    ensemble = VotingClassifier(
                        estimators=ensemble_models,
                        voting='soft',  # Olasılık tahminlerini kullan
                        n_jobs=-1
                    )
                    
                    # Ensemble modelini eğit
                    ensemble.fit(X_train_balanced, y_train_balanced)
                    
                    # Ensemble performansını değerlendir
                    ensemble_scores = cross_val_score(ensemble, X_train_balanced, y_train_balanced, 
                                                    cv=cv, scoring='balanced_accuracy')
                    ensemble_score = ensemble_scores.mean()
                    
                    print(f"Ensemble model capraz dogrulama skoru: {ensemble_score:.4f}")
                    
                    # Ensemble daha iyiyse, onu kullan
                    if ensemble_score > best_score:
                        print("Ensemble model, tekli modellerden daha iyi performans gosteriyor!")
                        best_model = ensemble
                        best_model_name = "Ensemble"
                        best_score = ensemble_score
                    else:
                        print(f"Tekli model ({best_model_name}) ensemble'dan daha iyi performans gosteriyor.")
                else:
                    print("Ensemble model olusturulamadi. predict_proba destegi olan yeterli model yok.")
            else:
                print("Ensemble model olusturulamadi. Yeterli sayida basarili model yok.")
        except Exception as e:
            print(f"Ensemble model olusturma hatasi: {str(e)}")
        
        # Son model ve sonuçları kaydet
        self.model = best_model
        self.best_model_name = best_model_name
        
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
        
        # Dengeli doğruluk ve diğer metrikler
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print(f"Hassasiyet (Precision): {precision:.4f}")
        print(f"Duyarlilik (Recall): {recall:.4f}")
        print(f"F1 Skoru: {f1:.4f}")
        
        # Sınıflandırma raporu
        print("\nSiniflandirma Raporu:")
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.cancer_types))]
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Karmaşıklık matrisi
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalizasyon
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Karmaşıklık matrisini görselleştir
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
        
        # Normalize edilmiş karmaşıklık matrisi
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Tahmin Edilen Etiket')
        plt.ylabel('Gercek Etiket')
        plt.title('Normalize Edilmis Karmasiklik Matrisi')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'normalized_confusion_matrix.png'))
        plt.close()
        
        # ROC eğrisi - Çok sınıflı için One-vs-Rest yaklaşımı
        try:
            # Model predict_proba metoduna sahipse
            if hasattr(model, 'predict_proba'):
                from sklearn.metrics import roc_curve, auc
                from itertools import cycle
                
                # Her bir sınıf için ROC eğrisi
                n_classes = len(self.cancer_types)
                y_test_bin = np.zeros((len(y_test), n_classes))
                for i in range(len(y_test)):
                    y_test_bin[i, y_test.iloc[i]] = 1
                
                y_score = model.predict_proba(X_test)
                
                # Her sınıf için ROC eğrisi hesapla
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                
                for i in range(n_classes):
                    if i < y_score.shape[1]:  # Bazı sınıflar tahmin edilemeyebilir
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                
                # ROC eğrilerini çiz
                plt.figure(figsize=(12, 10))
                
                # En çok örneği olan 10 sınıfı göster
                class_counts = Counter(y_test)
                top_classes = [class_idx for class_idx, _ in class_counts.most_common(10)]
                
                colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta'])
                
                for i, color in zip(top_classes, colors):
                    if i in roc_auc:
                        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                                label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--', lw=2)
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Egrileri (One-vs-Rest)')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(self.data_dir, 'roc_curves.png'))
                plt.close()
        except Exception as e:
            print(f"ROC egrisi cizim hatasi: {str(e)}")
        
        # Yanlış sınıflandırılan örnekleri incele
        misclassified_indices = np.where(y_test != y_pred)[0]
        misclassified_data = {
            'Sample ID': y_test.index[misclassified_indices],
            'True Cancer Type': self.label_encoder.inverse_transform(y_test.iloc[misclassified_indices]),
            'Predicted Cancer Type': self.label_encoder.inverse_transform(y_pred[misclassified_indices])
        }
        misclassified_df = pd.DataFrame(misclassified_data)
        
        # En sık karıştırılan sınıfları analiz et
        confusion_pairs = Counter(zip(misclassified_data['True Cancer Type'], 
                                    misclassified_data['Predicted Cancer Type']))
        
        print("\nEn fazla karistirilan kanser turleri:")
        for (true_type, pred_type), count in confusion_pairs.most_common(5):
            print(f"Gercek: {true_type}, Tahmin: {pred_type}, Sayi: {count}")
        
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
        
        try:
            if method == 'tsne':
                # t-SNE ile boyut indirgeme - perplexity parametresi önemli
                embedding = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X) // 5))
                X_embedded = embedding.fit_transform(X)
                title = 't-SNE ile Kanser Turlerinin Dagilimi'
            else:
                # PCA ile boyut indirgeme
                embedding = PCA(n_components=2)
                X_embedded = embedding.fit_transform(X)
                title = 'PCA ile Kanser Turlerinin Dagilimi'
            
            # Görselleştirme - daha şık
            plt.figure(figsize=(14, 10))
            
            # Benzersiz etiketler
            unique_labels = sorted(set(y))
            
            # Renk haritası
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            
            # Her sınıf için ayrı çiz
            for i, label in enumerate(unique_labels):
                idx = y == label
                plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], 
                            c=[cmap(i)], 
                            label=self.label_encoder.inverse_transform([label])[0],
                            alpha=0.7, s=50, edgecolors='k')
            
            plt.title(title)
            plt.xlabel('Bilesen 1')
            plt.ylabel('Bilesen 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, f'{method}_visualization.png'))
            plt.close()
        
        except Exception as e:
            print(f"{method} gorsellestirme hatasi: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_model(self, filename='ccle_cancer_classifier_model.pkl'):
        """
        Eğitilmiş modeli kaydet
        
        Args:
            filename: Model dosyasının adı
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
            'selected_features': self.selected_features,
            'cancer_types': self.cancer_types,
            'model_name': getattr(self, 'best_model_name', type(self.model).__name__),
            'feature_importance': self.feature_importance
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
            self.best_model_name = model_package.get('model_name', type(self.model).__name__)
            self.feature_importance = model_package.get('feature_importance', None)
            
            print(f"Model basariyla yuklendi: {model_path}")
            print(f"Model turu: {self.best_model_name}")
            print(f"Secili gen sayisi: {len(self.selected_features)}")
            print(f"Siniflandirilabilecek kanser turleri: {len(self.cancer_types)}")
            
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
            
            # Eksik genleri median değerleriyle doldur
            for gene in set(self.selected_features) - common_genes:
                sample_data[gene] = 0
        
        # Sadece seçilen genleri al
        sample_data = sample_data[self.selected_features]
        
        # Aykırı değerleri sınırla
        for col in sample_data.columns:
            series = sample_data[col]
            mean = series.mean()
            std = series.std()
            sample_data[col] = series.clip(lower=mean - 3*std, upper=mean + 3*std)
        
        # Ölçeklendir
        sample_scaled = self.scaler.transform(sample_data)
        
        # Tahmin yap
        prediction = self.model.predict(sample_scaled)[0]
        
        # Tahmin olasılıkları
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(sample_scaled)[0]
            probability = max(probabilities)
        else:
            probability = None
        
        # Etiket kodunu gerçek kanser türüne çevir
        cancer_type = self.label_encoder.inverse_transform([prediction])[0]
        
        return cancer_type, probability
        
    def run_pipeline(self, n_features=500, balance_data=True):
        
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

        self.create_gene_expression_heatmap()
        
        return best_model, accuracy
    
    def create_gene_expression_heatmap(self, top_n_genes=50):
        """En önemli genlerin kanser türlerine göre ekspresyon heatmap'ini oluşturur"""
        if self.feature_importance is None:
            print("Önce feature importance hesaplanmalı")
            return
        
        # En önemli N geni al
        top_genes = self.feature_importance['feature'].head(top_n_genes).tolist()
        
        # Her kanser türü için ortalama ekspresyon değerlerini hesapla
        cancer_exp_means = {}
        for cancer_type in self.cancer_types:
            cancer_idx = self.label_encoder.transform([cancer_type])[0]
            indices = self.labels[self.labels == cancer_idx].index
            cancer_exp_means[cancer_type] = self.exp_data.loc[indices, top_genes].mean()
        
        # DataFrame'e dönüştür
        heatmap_data = pd.DataFrame(cancer_exp_means).T
        
        # Z-score normalizasyonu
        from scipy import stats
        heatmap_data = heatmap_data.apply(stats.zscore, axis=0)
        
        # Heatmap çiz
        plt.figure(figsize=(15, 10))
        sns.clustermap(heatmap_data, cmap="coolwarm", figsize=(15, 10), 
                    dendrogram_ratio=(.1, .2), col_cluster=True, row_cluster=True)
        plt.title('Kanser Türlerine Göre Önemli Genlerin Ekspresyon Profili')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'cancer_gene_heatmap.png'))
        plt.close()

    def identify_cancer_specific_genes(self, save_results=True, top_n_genes=20):
        """
        Her kanser türü için ayırt edici genleri belirler ve sonuçları kaydeder
        
        Args:
            save_results: Sonuçların dosyalara kaydedilip kaydedilmeyeceği
            top_n_genes: Görselleştirme için her kanser türünde gösterilecek en önemli gen sayısı
            
        Returns:
            Kanser türlerine özgü genlerin bulunduğu sözlük
        """
        cancer_specific_genes = {}
        
        # Sonuçlar için klasör oluştur
        results_dir = os.path.join(self.data_dir, "cancer_specific_genes")
        if save_results:
            os.makedirs(results_dir, exist_ok=True)
        
        # Özet CSV için veri çerçevesi oluştur
        all_results = []
        
        # Tüm kanser türleri için döngü
        for cancer_type in self.cancer_types:
            print(f"Analyzing cancer type: {cancer_type}")
            
            # Etiket kodunu al
            try:
                cancer_code = self.label_encoder.transform([cancer_type])[0]
            except:
                # Eğer cancer_type bir string değilse, direkt kendisini kullan
                cancer_code = cancer_type
                cancer_type = self.label_encoder.inverse_transform([cancer_type])[0]
            
            # Bu kanser türüne ait indeksler
            cancer_indices = self.labels[self.labels == cancer_code].index
            # Diğer kanserler
            other_indices = self.labels[self.labels != cancer_code].index
            
            # Bu kanser türünün ve diğerlerinin ortalama ekspresyon değerleri
            cancer_mean = self.exp_data.loc[cancer_indices].mean()
            other_mean = self.exp_data.loc[other_indices].mean()
            
            # Farklılık skoru (fold change)
            fold_change = cancer_mean / other_mean.replace(0, 0.001)  # Sıfıra bölünmeyi önle
            
            # t-test ile p-değerleri hesapla
            from scipy import stats
            p_values = []
            for gene in self.exp_data.columns:
                t, p = stats.ttest_ind(
                    self.exp_data.loc[cancer_indices, gene],
                    self.exp_data.loc[other_indices, gene],
                    equal_var=False  # Welch's t-test
                )
                p_values.append(p)
            
            # Sonuçları birleştir
            result = pd.DataFrame({
                'gene': self.exp_data.columns,
                'fold_change': fold_change,
                'p_value': p_values,
                'log2_fold_change': np.log2(fold_change)
            })
            
            # İstatistiksel olarak anlamlı ve fold change'i yüksek genleri seç
            significant_genes = result[(result['p_value'] < 0.05) & (abs(result['log2_fold_change']) > 1)]
            significant_genes = significant_genes.sort_values('fold_change', ascending=False)
            
            # Sonuçları sözlüğe ekle
            cancer_specific_genes[cancer_type] = significant_genes
            
            # Özet dataframe için veri hazırla
            significant_genes['cancer_type'] = cancer_type
            all_results.append(significant_genes)
            
            if save_results:
                # Her kanser türü için CSV dosyasına kaydet
                csv_file = os.path.join(results_dir, f"{cancer_type.replace(' ', '_')}_specific_genes.csv")
                significant_genes.to_csv(csv_file, index=False)
                print(f"Saved {len(significant_genes)} genes for {cancer_type} to {csv_file}")
                
                # En önemli genlerin görselleştirilmesi
                if len(significant_genes) > 0:
                    # Pozitif ve negatif fold change'leri ayır
                    upregulated = significant_genes[significant_genes['fold_change'] > 1].head(top_n_genes)
                    downregulated = significant_genes[significant_genes['fold_change'] < 1].tail(top_n_genes)
                    
                    # Görselleştirme için genleri birleştir
                    plot_genes = pd.concat([upregulated, downregulated])
                    
                    if len(plot_genes) > 0:
                        plt.figure(figsize=(12, 8))
                        
                        # Log2 fold change'e göre sırala
                        plot_genes = plot_genes.sort_values('log2_fold_change')
                        
                        # Barplot oluştur
                        colors = ['red' if fc > 0 else 'blue' for fc in plot_genes['log2_fold_change']]
                        plt.barh(plot_genes['gene'], plot_genes['log2_fold_change'], color=colors)
                        
                        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        plt.xlabel('Log2 Fold Change')
                        plt.ylabel('Gen')
                        plt.title(f'{cancer_type} için Ayırt Edici Genler')
                        plt.tight_layout()
                        
                        # Kaydet
                        plt.savefig(os.path.join(results_dir, f"{cancer_type.replace(' ', '_')}_specific_genes.png"))
                        plt.close()
                        print(f"Saved gene visualization for {cancer_type}")
        
        if save_results and all_results:
            # Tüm sonuçları birleştir ve kaydet
            all_specific_genes = pd.concat(all_results)
            all_specific_genes.to_csv(os.path.join(results_dir, "all_cancer_specific_genes.csv"), index=False)
            print(f"Saved combined results to {os.path.join(results_dir, 'all_cancer_specific_genes.csv')}")
            
            # Özet rapor oluştur
            summary_file = os.path.join(results_dir, "cancer_specific_genes_summary.txt")
            with open(summary_file, 'w') as f:
                f.write("Kanser Türlerine Özgü Gen Analizi Özeti\n")
                f.write("=====================================\n\n")
                
                for cancer_type, genes in cancer_specific_genes.items():
                    f.write(f"Kanser Türü: {cancer_type}\n")
                    f.write(f"Ayırt Edici Gen Sayısı: {len(genes)}\n")
                    
                    if len(genes) > 0:
                        f.write("En Çok Upregulated Genler (Fold Change):\n")
                        for idx, row in genes[genes['fold_change'] > 1].head(5).iterrows():
                            f.write(f"  - {row['gene']}: {row['fold_change']:.2f} (p-value: {row['p_value']:.4f})\n")
                        
                        f.write("En Çok Downregulated Genler (Fold Change):\n")
                        for idx, row in genes[genes['fold_change'] < 1].head(5).iterrows():
                            f.write(f"  - {row['gene']}: {row['fold_change']:.2f} (p-value: {row['p_value']:.4f})\n")
                    
                    f.write("\n")
                
                f.write("NOT: Tüm sonuçlar 'all_cancer_specific_genes.csv' dosyasında bulunabilir.\n")
            
            print(f"Saved summary report to {summary_file}")
            
            # Heatmap oluştur - tüm kanser türleri için ortak önemli genleri göster
            try:
                # En çok kanser türünde görülen genleri bul
                gene_counts = all_specific_genes['gene'].value_counts().head(50)
                common_important_genes = gene_counts.index.tolist()
                
                if common_important_genes:
                    # Her kanser için bu genlerin ortalama ekspresyonunu hesapla
                    heatmap_data = {}
                    for cancer_type in self.cancer_types:
                        try:
                            cancer_code = self.label_encoder.transform([cancer_type])[0]
                        except:
                            cancer_code = cancer_type
                            cancer_type = self.label_encoder.inverse_transform([cancer_type])[0]
                        
                        cancer_indices = self.labels[self.labels == cancer_code].index
                        gene_means = self.exp_data.loc[cancer_indices, common_important_genes].mean()
                        heatmap_data[cancer_type] = gene_means
                    
                    # DataFrame'e dönüştür
                    heatmap_df = pd.DataFrame(heatmap_data).T
                    
                    # Z-score normalizasyonu
                    from scipy import stats
                    heatmap_df_norm = heatmap_df.apply(stats.zscore, axis=0)
                    
                    # Heatmap çiz
                    plt.figure(figsize=(15, 12))
                    sns.clustermap(heatmap_df_norm, cmap="coolwarm", figsize=(15, 12), 
                                dendrogram_ratio=(.1, .2))
                    plt.savefig(os.path.join(results_dir, "cancer_gene_expression_heatmap.png"))
                    plt.close()
                    print(f"Saved heatmap to {os.path.join(results_dir, 'cancer_gene_expression_heatmap.png')}")
            except Exception as e:
                print(f"Heatmap oluşturma hatası: {str(e)}")
        
        return cancer_specific_genes

# Kullanım örneği
if __name__ == "__main__":
    # Sınıflandırıcıyı başlat
    classifier = CCLECancerClassifier(data_dir="./ccle_data")
    
    # Manuel indirilen dosyaları kullan
    classifier.expression_file = "./ccle_data/CCLE_expression.csv"
    classifier.annotation_file = "./ccle_data/sample_info.csv"
    
    # Tüm işlem hattını çalıştır
    model, accuracy = classifier.run_pipeline(
        n_features=500,      # Daha az gen kullanarak overfitting azalt
        balance_data=True,   # Veri dengesizliğini düzelt
    )
    
    print("\n=== Proje Tamamlandi ===")
    print(f"Model kaydedildi. Test dogrulugu: {accuracy:.4f}")