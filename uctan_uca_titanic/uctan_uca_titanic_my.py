##################
# Kütüphaneler
##################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

###########################
# FONKSİYONLAR
###########################

# 1-) Aykırı değerler için limit belirledik.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# 2-) Kolonlar içinde bu limitleri aşan aykırı değerler var mı diye sormak.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# 3-) Sinsirellaları tespit etme.
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

# 4-) Aykırı değerlerin index bilgilerine ulaşmai
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:    # eğer aykırı değer sayısı 10'dan büyükse
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())    # Bu aykırı değerlere head at ve görmem için yazdır.
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])           # eğer aykırı değer sayısı 10'dan az ise hepsini yazdır

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        # eğer aykırı değerlerin index bilgisini istiyorsan ön tanımlı index=False bilgisini index=True yap.
        # Böylece aykırı değerlerin index bilgisini return edeceksin.
        return outlier_index

# 5-) Aykırı değerleri silmek gerekiyorsa.
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers
cat_cols, num_cols, cat_but_car = grab_col_name(df)

# 6-) Aykırı değerleri baskılamamız gerekiyorsa.
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit    # alt limitten daha aşağıda olanları alt limite eşitle.
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit      # üst limitten daha yukarda olanaları üst limite eşitle

# 7-) Eksik değerler ve oranlar tablosu
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

# 8-) Eksik değerlerin ve eksik olmayanların hedef değişkene etkisi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)   # eksik değer olanalara 1 olmayanlara 0 ver.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end= "\n\n")

# 9-) Binary Encoding(büyüklük anlamı taşır)
def label_encoding(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# 10-) One-Hot Encoder (get-dummies)
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# 11-) Kategorik değişken içinde kaç farklı sınıf var, sınıfların içinde kaç adet değer var?
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# 12-) Rare analizi, hangi kolonlarda rare olabilecek sınıf var?
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

# 13-) Rare olabilecekleri rare encode etmek.
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# 14-) Sayısal değişkenlerin kaç sınıfı var, sınıflarda kaç değer var? (bunu standartlaştırmadan sonra yapmak gerek)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# elde edilebilecek tüm yeni değişkenleri oluşturalım. Her değişken için bakılmalı.

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()
df.shape   # (891, 22) artık 22 değişkenimiz var.

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

num_cols = [col for col in num_cols if "PASSENGERID" not in col]


#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

# AGE True
# FARE True
# NEW_NAME_COUNT True
# NEW_AGE_PCLASS True  Aykırı değerler var.


# aykırı değerleri tıraşlayalım.

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit    # alt limitten daha aşağıda olanları alt limite eşitle.
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit      # üst limitten daha yukarda olanaları üst limite eşitle

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# AGE False
# FARE False
# NEW_NAME_COUNT False
# NEW_AGE_PCLASS False  aykırı değerlerden kurtulduk.

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

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

missing_values_table(df, True)     # Age değişkenindeki eksiklikler sonradan oluşturduğumuz değişkenlerde de var.
# Önce Cabin değişkenini uçuralım. çünkü kabinle alakalı anlamlı bir değişken ürettik zaten

#                 n_miss  ratio
# CABIN              687  77.10
# AGE                177  19.87
# NEW_AGE_PCLASS     177  19.87
# NEW_AGE_CAT        177  19.87
# NEW_SEX_CAT        177  19.87
# EMBARKED             2   0.22

df.drop("CABIN", inplace=True, axis=1)

# Name ve ticket değişkenini de uçuralım. Name değişkeninden zaten yeni bir değişken oluşturduk.

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

# yaş değişkenindeki eksiklikleri new title a göre dolduralım

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# şimdi yaşa bağlı oluşturduğumuz tüm değişkenleri tekrar oluşturalım

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

missing_values_table(df, True) # sadece embarked kaldıç
#           n_miss  ratio
# EMBARKED       2   0.22
# Out[37]: ['EMBARKED']

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

missing_values_table(df, True)  # eksik değer hiç kalmadı.

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols # ['SEX', 'NEW_IS_ALONE']

def label_encoding(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoding(df, col)

#############################################
# 5. Rare Encoding
#############################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SURVIVED", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

rare_analyser(df, "SURVIVED", cat_cols)  # 0.01'in altında kalanların hepsi rare sınıfı altında birleşti.

# NEW_TITLE : 5
#         COUNT     RATIO  TARGET_MEAN
# Master     40  0.044893     0.575000
# Miss      182  0.204265     0.697802
# Mr        517  0.580247     0.156673
# Mrs       125  0.140292     0.792000
# Rare       27  0.030303     0.444444


#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols, True)

df.head()
df.shape  # (891, 52)

# bir sürü yeni değişken oluştu ama bazı değişkenlerin %99 u 1 %1 sıfır ise, bir anlam taşımaz.
# Tekrar rare analiz yapıp bakalım

cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

# oluşturduğumuz bazı değişkenler anlamlı bilgiler taşımıyor. Bunları topluca görelim;

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

useless_cols # ['SIBSP_5','SIBSP_8','PARCH_3','PARCH_4','PARCH_5','PARCH_6','NEW_NAME_WORD_COUNT_9','NEW_NAME_WORD_COUNT_14','NEW_FAMILY_SIZE_8','NEW_FAMILY_SIZE_11']

# bu değişkenler bilgi taşımıyor, silinebilir.

df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape  # (891, 42) artık verimiz hazır.

#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)   # 0.8097014925373134

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)   #  0.7090909090909091


# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

# buradaki grafikte hangi değişkenin ne kadar etkili olduğunu görüyoruz.

