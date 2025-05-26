# -*- coding: utf-8 -*-
"""Kualitas_Apel_sion.ipynb

# **Predictive Analytics: Kualitas Apel**

- **Nama:** Sion Saut Parulian Pardosi
- **Email:** spardosi12@gmail.com
- **Email Dicoding:** mc114d5y1919@student.devacademy.id
- **ID Dicoding:** MC114D5Y1919


---


## **Deskripsi Proyek**

### **Deskripsi Latar Belakang Proyek Prediksi Kualitas Apel dengan Machine Learning**


Proyek ini bertujuan untuk mengembangkan model machine learning yang dapat memprediksi kualitas apel dengan lebih akurat dan efisien. Saat ini, penentuan kualitas apel masih dilakukan secara manual, yang memakan waktu, tenaga, dan rentan terhadap kesalahan. Hal ini menyebabkan kerugian bagi petani dan distributor, serta memberikan produk yang tidak sesuai dengan harapan konsumen. Model prediksi kualitas apel dapat membantu mengatasi permasalahan ini dengan memberikan solusi yang lebih akurat, efisien, dan transparan.

## 1. Import Library yang Dibutuhkan
"""

#Import Load data Library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
import numpy as np

# Import train test split
from sklearn.model_selection import train_test_split
# Import Minmaxscaler
from sklearn.preprocessing import MinMaxScaler
#Import Model
from sklearn.neighbors import KNeighborsClassifier # --> KNN
from sklearn.ensemble import RandomForestClassifier # --> RF
from sklearn.metrics import accuracy_score # --> matrix accuracy
from sklearn.svm import SVC # --> SVM
from sklearn.naive_bayes import BernoulliNB # --> Naive bayes
from sklearn.ensemble import ExtraTreesClassifier # --> Extra Trees Classifier


"""## 2. Data Understanding  


Data Understanding merupakan proses memahami informasi dalam data dan menentukan kualitas dari data tersebut.

### 2.1 Data Loading



Data Loading merupakan tahap untuk memuat dataset yang akan digunakan agar dataset lebih mudah dipahami.
Untuk informasi datasets ini telah di *bersihan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula.


<br>


**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | Apple Quality |
| Source | [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality/data) |
| Maintainer | [Nidula Elgiriyewithana ⚡](https://www.kaggle.com/nelgiriyewithana) |
| License | Other (specified in description) |
| Visibility | Publik |
| Tags | Computer Science, Education, Food, Data Visualization, Classification, Exploratory Data Analysis |
| Usability | 10.00 |
"""

df = pd.read_csv(r'D:\PROJECT\DBS DICODING\machine learning terapan\Proyek-Machine-Learning-Dicoding-Predictive-Analytics-Kualitas-Apel-main\apple_quality.csv')

"""# **2.2 Exploratory Data Analysis (EDA)**

Exploratory data analysis merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

## **2.2.1 EDA - Deskripsi Variabel**
"""

df

"""Dari dataframe di atas kita dapat melihat bahwa pada dataset ini terdapat 9 kolom. Diantaranya:

- `A_id` : Pengidentifikasi unik untuk setiap buah
- `Size` : Ukuran buah
- `Weight` : Berat buah
- `Sweetness` : Tingkat kemanisan buah
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah
-` Juiciness` : Tingkat kesegaran buah
- `Ripeness` : Tahap kematangan buah
- `Acidity` : Tingkat keasaman buah
- `Quality` : Kualitas buah secara keseluruhan
"""

df.drop("A_id",axis=1,inplace=True)

"""Dikarenakan kolom `A_id` tidak mempengaruhi model maka akan di drop / dihapus.




"""

df.info()

"""Dari eksekusi method `df.info()` terdapat:

- Terdapat 6 kolom numerik dengan tipe data float64 yaitu: Size, Weight, Sweetness, Crunchiness, Juiciness dan Ripeness.
- Terdapat 2 kolom dengan tipe data object yaitu: Acidity dan Quality.

Namun pada data aslinya kolom ` Acidity` adalah bertipe float64, yang nantinya akan kita rubah.
"""

df.describe()

"""Fungsi `describe()` memberikan informasi statistik pada masing-masing kolom, antara lain:

- `Count` adalah jumlah sampel pada data.
- `Mean` adalah nilai rata-rata.
- `Std` adalah standar deviasi.
- `Min` yaitu nilai minimum setiap kolom.
- `25%` adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- `50%` adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
-` 75%` adalah kuartil ketiga.
- `Max` adalah nilai maksimum.
"""

df.shape

"""Dari eksekusi method` df.shape` Terlihat:
<br>

| Jumlah Baris | Jumlah Kolom |
| ------ | ------ |
| 4001 | 8 |


<br>

## **2.2.2 EDA - Menangani Missing Value dan Outliers**
"""

df.duplicated().sum()

"""Melihat apakah terdapat data yang terduplikat."""

df.Quality.value_counts(normalize=True)

df.isnull().sum()

data_miss = df[df.isnull().any(axis=1)]
data_miss

"""Dapat dilihat terdapat missing value yang mana akan kita hapus."""

df.dropna(inplace=True)
df.isnull().sum().sum()

df.describe()

df["Acidity"] = df["Acidity"].astype("float64")

"""Merubah tipe data kolom `Acidity` menjadi data float64."""

df.info()

"""Dapat kita lihat:
- Jumlah data` Float64` ada 7 dan `object `ada 1.
"""

df.shape

"""Jumlah datasets menjadi `4000` dikarenakan kita telah menghapus missing value.

**Visualisasi Outlier**
"""

df_outlier=df.select_dtypes(exclude=['object'])
for column in df_outlier:
        plt.figure()
        sns.boxplot(data=df_outlier, x=column)

"""*Menghapus outliers yang ada pada dataset*  


Pada kasus ini, kita akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, menangani outliers dengan teknik IQR method.


```
IQR = Inter Quartile Range
IQR = Q3 - Q1
```


"""

# Ambil hanya kolom numerik
df_numeric = df.select_dtypes(include=[np.number])

# Hitung Q1, Q3, dan IQR
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Filter outlier hanya berdasarkan kolom numerik
df_clean = df[~((df_numeric < (Q1 - 1.5 * IQR)) | (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]

"""Jumlah Datasets setalah kita hapus Outlier: `3790, 8`

## **2.2.3 EDA - Univariate Analysis**
"""

df.hist(bins=50, figsize=(20,15))
plt.show()

"""## **2.2.4 EDA - Multivariate Analysis**"""

sns.pairplot(df, diag_kind = 'kde')

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title(f"Matriks Korelasi untuk Fitur Numerik ", size=20)

"""# **3. Data Preparation**

Data Preparation merupakan tahap untuk mempersiapkan data sebelum masuk ke tahap pembuatan model Machine Learning.

## **3.1 Data Clening**
"""

df.Quality = (df.Quality == "good").astype(int)  # good:1 , bad:0

x = df.drop("Quality",axis=1)
y = df.Quality

x.shape,y.shape

"""## **3.2 Train-Test-Split**"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=60)

print(f'Total datasets: {len(x)}')
print(f'Total data Latih: {len(x_train)}')
print(f'Total data Uji: {len(x_test)}')

"""## **3.3 Normalisasi**"""

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

"""# **4. Model Development**

## **Lazy Predict Library**

**LazyPredict** adalah pustaka Python yang memudahkan proses pemilihan model machine learning. Ia melakukan ini dengan secara otomatis mengevaluasi dan membandingkan berbagai algoritma pembelajaran mesin pada kumpulan data.

- Keuntungan menggunakan LazyPredict:
 * **Cepat dan efisien**: LazyPredict dapat dengan cepat mengevaluasi dan membandingkan banyak model, menghemat waktu dan tenaga.
 * **Mempermudah identifikasi model potensial**: Alih-alih mencoba berbagai model secara manual, LazyPredict membantu menemukan model yang berpotensi berkinerja baik pada data.
 * **Cocok untuk analisis awal dan pembuatan prototipe**: LazyPredict memudahkan untuk memulai dengan proyek machine learning dengan cepat tanpa terjebak dalam detail pemilihan model.
"""

!pip install lazypredict

from lazypredict.Supervised import LazyClassifier
clf = LazyClassifier()
models,predicts = clf.fit(x_train,x_test,y_train,y_test)
print(models.sort_values(by="Accuracy",ascending=False))

"""**Visualisasi Model LazyPredict**"""

temp = models.sort_values(by="Accuracy",ascending=True)
plt.figure(figsize=(10, 8))
plt.barh(temp.index,temp["Accuracy"])
plt.show()

models = pd.DataFrame(index=['accuracy_score'],
                      columns=['KNN', 'RandomForest', 'SVM', 'Naive Bayes','Extra trees classifier'])

"""## **4.1 KNN (K-Nearest Neighbor)**"""

model_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
model_knn.fit(x_train, y_train)

knn_pred = model_knn.predict(x_test)
models.loc['accuracy_score','KNN'] = accuracy_score(y_test, knn_pred)

"""## **4.2 Random Forest**"""

model_rf = RandomForestClassifier(max_depth= 20)
model_rf.fit(x_train, y_train)

rf_pred = model_rf.predict(x_test)
models.loc['accuracy_score','RandomForest'] = accuracy_score(y_test, rf_pred)

"""## **4.3 Support Vector Classifier**"""

model_svc = SVC()
model_svc.fit(x_train, y_train)

svc_pred = model_svc.predict(x_test)
models.loc['accuracy_score','SVM'] = accuracy_score(y_test, svc_pred)

"""### **4.4 Naive Bayes**"""

model_nb = BernoulliNB()
model_nb.fit(x_train, y_train)

nb_pred = model_nb.predict(x_test)
models.loc['accuracy_score','Naive Bayes'] = accuracy_score(y_test, nb_pred)

"""### **4.5 Extra trees classifier**"""

model_etc = ExtraTreesClassifier(n_estimators=100, max_depth= 10,n_jobs= 2,random_state= 100)
model_etc.fit(x_train, y_train)

etc_pred = model_etc.predict(x_test)
models.loc['accuracy_score','Extra trees classifier'] = accuracy_score(y_test, etc_pred)

"""# **5. Evaluasi Model**

## **5.1 Score Model**
"""

print(models)

"""## **5.2 plot Model**"""

plt.bar('KNN', models['KNN'])
plt.bar('RandomForest', models['RandomForest'])
plt.bar('SVM', models['SVM'])
plt.bar('Naive Bayes', models['Naive Bayes'])
plt.bar('Extra Trees', models['Extra trees classifier'])
plt.title("Perbandingan Akurasi Model");
plt.xlabel('Model');
plt.ylabel('Akurasi');
plt.show()