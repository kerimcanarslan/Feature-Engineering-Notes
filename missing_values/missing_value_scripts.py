##################
# MISSING VALUES (EKSIK DEGERLER)
##################


#################
# EKSIK DEGERLERIN YAKALANMASI
#################

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

df.isnull()
df.isnull().any()      # Age, Cabin ve Embarked değişkeni True döndü
df.isnull().values.any()

df.isnull().sum()      # hangi değişkende kaç eksik değer var. bunların toplamını getir.

df.notnull().sum()     # hangi değişkenlerde ne kadar dolu hücre var.


# herhangi bir satırda eksik değer varsa bunları topluca görelim

df[df.isnull().any(axis=1)]   # eksik değer bulunan tüm satırlar.

# tamamen dolu olan satırlar

df[df.notnull().all(axis=1)]   # tüm değişkend değerleri dolu olan satırlar


# azalan şeilde sıralayalım

df.isnull().sum().sort_values(ascending=False)  #Cabin 687, Age 177, Embarked 2

# değişkenlerin yüzde kaçı eksik değer ona bakalım

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)  # Cabin 77.104377, Age 19.865320, Embarked 0.224467

# Sadece eksik değerler olan kolonları belirlemek için bir list. comp. yzalım

na_col = [col for col in df.columns if df[col].isnull().sum() > 0]

na_col # ['Age', 'Cabin', 'Embarked']


# Şimdi tüm bunlar için script hazırlayalım

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # eksik değer içeren değişkenleri tuttuk.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # içindeki eksik değer sayılarına göre büyükten küçüğe sıraladık ve n_mis adında yeni bir df oluşturduk.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # yüzdelik olarak oranlarını da ratio adında bir df'e attık.

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
# Böyle bir çıktı alırız.

#          n_miss  ratio
#Cabin        687  77.10
#Age          177  19.87
#Embarked       2   0.22

missing_values_table(df, na_name=True)
# Ön tanımlı değeri True yaparsak

#          n_miss  ratio
#Cabin        687  77.10
#Age          177  19.87
#Embarked       2   0.22
#Out[24]: ['Age', 'Cabin', 'Embarked']


#########################
# EKSIK DEGER PROBLEMLERINI COZME
#########################

# 1-) Hızlıca silebiliriz.,

df.dropna().shape   # yeniden atama yapılmalıdır. burada sadece gözlemledik


# 2-) Basit atama yöntemlerini kullanabiliriz (mean, median, vs)

df["Age"].fillna(df["Age"].mean()).isnull().sum()   # yeniden atama yapılmalıdır. burada sadece gözlemledik

## bunu tüm df'e apply ile uygulayaılım

# sayısal değişkenler için ortlama ya da median almak mantıklıdır
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)     # axis=0 O değişkenin Tüm sütundaki değerleinin ortlaması için

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)

missing_values_table(dff, na_name=True)
#          n_miss  ratio
#Cabin        687  77.10
#Embarked       2   0.22
#Out[37]: ['Cabin', 'Embarked']

# Artık Age değişkeninde eksik değer yok gördülüğü gibi. Ama kategorik değişkenlerde hala eksik değerler var.


# kategorik değişkneler için ise mod almak mantıklıdır.

df["Embarked"].fillna(df["Embarked"].mode()[0])

# eğer öncesinde sinsirellalar tespit edilmemiş ise bu şekilde yaparız.
# Tespit edip astype yaptıysak da sadece x.dtype == 'O' kullansakta olur

dff = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0)

missing_values_table(dff, na_name=True)
# Burada da embarked değişkeninde eksikliker giderildi

#        n_miss  ratio
# Cabin     687  77.10
# Age       177  19.87
# Out[40]: ['Age', 'Cabin']



#######################
# KATEGORİK DEĞİŞKEN KIRILIMINDA DEĞER ATAMA
#######################

# yaş değişkeninde eksik değerlere ortalamayı atadık. Ama kadın ve erkek yaş ortalamaları arasında bir fark var

df["Age"].mean()   # 29.69911764705882  biz bunu atadık.

df.groupby("Sex")["Age"].mean()  # Fakat cinsiyet kırılımında ortalamalar değişik.
# female    27.915709
# male      30.726645


df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# burada transform kullanarak bu sorunu çözdük. Eksik olan değerin cinsiyeti kadınsa kadın yaş ortalaması
# erkekse erkek yaş ortalaması yeni değer olarak verildi.


# trasnform kullanmadan daha low level yaparsak?

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

# yaparak da atayabiliriz. ama bunu tüm cinsiyetler için tek tek yapmka gerekir. Transform bunu tek seferde hallederç


# 3-) Tahmine Dayalı Atama ile Doldurma

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

def grab_col_name(dataframe, cat_th= 10, car_th=20):
    # 1-) Kategorik Değişkenleri seçelim
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # Sayısal gibi görünen ama Kategorik olan değişkenleri seçelim
    num_but_cat =[col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                  dataframe[col].dtypes != "O"]
    # Kategorik gibi görünen ama Sayısal(Kardinal) olan değişkenleri seçelim
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Şimdi Kategorik Kolonları son haline getirelim
    cat_cols = cat_cols + num_but_cat   # ikisini birleştirelim
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # Kategorik olupta sayısal olanları  çıkaralım

    # 2-) Sayısal(numerik) Değişkenleri Seçelim
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # Sayısal olanalrı seçelim
    num_cols = [col for col in num_cols if col not in num_but_cat]  # Sayısal görünüp kat. olanları çıkaralım

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    # Biz genelde sayısal değişkenler üzerinde çalışacağımız için num_cols ve cat_but_car bize lazım olur.
    # Ama cat_cols'u da return edelim. num_but_cat'i zaten cat_cols'un içine attık. Ayrıca onu return etmeye gerek yok.
    # cat_cols + num_cols + cat_but_car toplamı toplam değişken sayısını verir.
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

### DİKKAT MAKİNE ÖĞRENMESİ

# get_dummies sadece kategorik değişkenelre bakar.

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)   # ilkini at, ikincisini tut. ikincisine 1 de, ilkine 0
# 2 sınıfa bölünmüş kategorik değişkenleri numerik olarak ifade etmek işlemidir. Mesela sex 1 ve o olarakyazılır
dff.head()

# değişkenlerin standartlaştırılması

scaler = MinMaxScaler()   # 0-1 arasında min max scala belirler
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()   # tüm değer 0-1 arasında oldu.

# şimdi eksik değerlerin tahmin edilip bunun içine doldurulması içim;

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)  # en aykın 5 komşusuna bakar ve onalrın ortalamasını alır

dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()   # boş değerlere komşularına bakarak 0-1 arasında bir değer atadı. eksik değerler doldu.

# ama biz 0-1 arası değerleri değil normal yaş değerlerini görmek istiyotuz.

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)   # standartlaştırmayı geri dönüştürme
dff.head()   # artık normal değerleri aldı.

# doldurduğum değerleri görmek istiyorum.

df["age_imputed_knn"] = dff["Age"]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]  # aşağıda ilk 3

#      Age  age_imputed_knn
# 5    NaN             47.8
# 17   NaN             37.6
# 19   NaN             12.2



#####################
# GELİŞMİŞ ANALİZLER  (EKSİK VERİNİN YAPISINI İNCELEYELİM)
#####################

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

msno.bar(df)    # değişkenlerdeki toplam doluluk
plt.show()

msno.matrix(df)   # değişkenlerdeki eksik değerleri beyaz boyar. dolular siyah boyar. karşılaştırmak için iyi bir araç
plt.show()

msno.heatmap(df)    # eksik değerlerin korelasyonunu verir. 2 değişkendeki eksik değerlerin birbiri ile korelasyonu
plt.show()


#######################
# EKSİK DEĞERLERİN BAĞIMLI DEĞİŞKEN İLE İLİŞKİSİNİN ANALİZİ
#######################

na_cols = missing_values_table(df, na_name=True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)   # eksik değer olanalara 1 olmayanlara 0 ver.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end= "\n\n")


missing_vs_target(df, "Survived", na_cols)

#              TARGET_MEAN  Count
# Age_NA_FLAG
# 0               0.406162    714           # age değişkeni eksik olmayanların ölme ihtimali
# 1               0.293785    177           # age değişkeni eksik olanların ölme ihtimali
#                TARGET_MEAN  Count
# Cabin_NA_FLAG
# 0                 0.666667    204         # cabin değişkeni eksik olmayanların ölme ihtimali
# 1                 0.299854    687         # cabin değişkeni eksik olanların ölme ihtimali
#                   TARGET_MEAN  Count
# Embarked_NA_FLAG
# 0                    0.382452    889
# 1                    1.000000      2



############# KISACA NELER YAPTIK

# 1-) Eksik değerlerin bulunduğu kolonları, kaç eksik değer olduğu ve yüzdelik oranları

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # eksik değer içeren değişkenleri tuttuk.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # içindeki eksik değer sayılarına göre büyükten küçüğe sıraladık ve n_mis adında yeni bir df oluşturduk.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # yüzdelik olarak oranlarını da ratio adında bir df'e attık.

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, na_name=True)

#           n_miss  ratio
# Cabin        687  77.10
# Age          177  19.87
# Embarked       2   0.22
# Out[25]: ['Age', 'Cabin', 'Embarked']

# 2-) Eksik değerlere ne yapılacağına karar verdik. Sil, atama yap ya da tahmin et

# 3-) Eksik değerlerin birbiri ile olan ilişkilerini analiz etmeye çalıştık.

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)   # eksik değer olanalara 1 olmayanlara 0 ver.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end= "\n\n")


missing_vs_target(df, "Survived", na_cols)

#              TARGET_MEAN  Count
# Age_NA_FLAG
# 0               0.406162    714           # age değişkeni eksik olmayanların ölme ihtimali
# 1               0.293785    177           # age değişkeni eksik olanların ölme ihtimali
#                TARGET_MEAN  Count
# Cabin_NA_FLAG
# 0                 0.666667    204         # cabin değişkeni eksik olmayanların ölme ihtimali
# 1                 0.299854    687         # cabin değişkeni eksik olanların ölme ihtimali
#                   TARGET_MEAN  Count
# Embarked_NA_FLAG
# 0                    0.382452    889
# 1                    1.000000      2
