

######################
# LABEL ENCODİNG (BINARY)
######################

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

df["Sex"].value_counts()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]   # binary haline çevirdik. Alfabetik sırada hangisi önce geliyorsa o sıfır değerini alır
le.inverse_transform([0, 1])[0:5]  # yaptığımız işlemi geri alır, aynı zamanda 0 ve 1'in hangi değerler olduğunu verir.


# Scriptini hazırlayalım

def label_encoding(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# binary olan değişkenleri bulalım

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

for col in binary_cols:
    label_encoding(df, col)

df["Sex"]


df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/application_train.csv")

df.shape   # 122 değişken var. bu veri setine uygulayalım

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and df[col].nunique() == 2]

binary_cols  # ['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE'] 4 değişken geldi

df[binary_cols].head()

#   NAME_CONTRACT_TYPE FLAG_OWN_CAR FLAG_OWN_REALTY EMERGENCYSTATE_MODE
# 0         Cash loans            N               Y                  No
# 1         Cash loans            N               N                  No
# 2    Revolving loans            Y               Y                 NaN
# 3         Cash loans            N               Y                 NaN
# 4         Cash loans            N               Y                 NaN

for col in binary_cols:
    label_encoding(df, col)

df[binary_cols].head()

#    NAME_CONTRACT_TYPE  FLAG_OWN_CAR  FLAG_OWN_REALTY  EMERGENCYSTATE_MODE
# 0                   0             0                1                    0
# 1                   0             0                0                    0
# 2                   1             1                1                    2
# 3                   0             0                1                    2
# 4                   0             0                1                    2



######################
# ONE HOT ENCODİNG
######################

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

df.head()
df["Embarked"].value_counts()  # S:644 C:168 Q:77  toplamda 3 farklı sınıf var. 2'den fazla olduğu için One-Hot encoding

pd.get_dummies(df, columns=["Embarked"]).head()

#    PassengerId  Survived  Pclass  ... Embarked_C Embarked_Q  Embarked_S
# 0            1         0       3  ...          0          0           1
# 1            2         1       1  ...          1          0           0
# 2            3         1       3  ...          0          0           1
# 3            4         1       1  ...          0          0           1
# 4            5         0       3  ...          0          0           1

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# drop first alfabetik sırada ilk geleni almaz. diğer kalan sınıflara encode etmeye başlar.;

#    PassengerId  Survived  Pclass  ... Cabin Embarked_Q  Embarked_S
# 0            1         0       3  ...   NaN          0           1
# 1            2         1       1  ...   C85          0           0
# 2            3         1       3  ...   NaN          0           1
# 3            4         1       1  ...  C123          0           1
# 4            5         0       3  ...   NaN          0           1

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()  # Her zaman kullanmayız

# eksik değer varsa onun içinde bir kolon oluşturur

#    PassengerId  Survived  Pclass  ... Embarked_Q Embarked_S  Embarked_nan
# 0            1         0       3  ...          0          1             0
# 1            2         1       1  ...          0          0             0
# 2            3         1       3  ...          0          1             0
# 3            4         1       1  ...          0          1             0
# 4            5         0       3  ...          0          1             0


# DİKKAT: sadece 2 sınıfı olan bir değişkende one-hot encoding drop_first ile kullanılırsa ki
# hep öyle kullanacağız. Bu label encoder olur. Farklı değişkenleri bir arada one-hot encodinge sokabiklriiz

pd.get_dummies(df, columns=["Embarked", "Sex"], drop_first=True).head()

# Hem hembarked'ın ilk sınıfını silerek hem de sex'in ilk sınıfı silerek encode etti.

#    PassengerId  Survived  Pclass  ... Embarked_Q  Embarked_S  Sex_male
# 0            1         0       3  ...          0           1         1
# 1            2         1       1  ...          0           0         0
# 2            3         1       3  ...          0           1         0
# 3            4         1       1  ...          0           1         0
# 4            5         0       3  ...          0           1         1


# Script haline getirelim

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

# cat_cols, num_cols, cat_but_car = grab_col_name(df) # Kullanabiliriz ama daha da irdeleyelim

ohe_cols = [col for col in df.columns if  10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols, drop_first=True)

#      PassengerId  Survived                                               Name     Sex    Age              Ticket      Fare            Cabin  Pclass_2  Pclass_3  SibSp_1  SibSp_2  SibSp_3  SibSp_4  SibSp_5  SibSp_8  Parch_1  Parch_2  Parch_3  Parch_4  Parch_5  Parch_6  Embarked_Q  Embarked_S
# 0              1         0                            Braund, Mr. Owen Harris    male  22.00           A/5 21171    7.2500              NaN         0         1        1        0        0        0        0        0        0        0        0        0        0        0           0           1
# 1              2         1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.00            PC 17599   71.2833              C85         0         0        1        0        0        0        0        0        0        0        0        0        0        0           0           0
# 2              3         1                             Heikkinen, Miss. Laina  female  26.00    STON/O2. 3101282    7.9250              NaN         0         1        0        0        0        0        0        0        0        0        0        0        0        0           0           1
# 3              4         1       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.00              113803   53.1000             C123         0         0        1        0        0        0        0        0        0        0        0        0        0        0           0           1


######################
# RARE ENCODİNG ( BJK: 200 GS: 250 FB: 300 TS: 1 GB:2 TV:3 gibi bir veri setinde TS GB ve TV'de çok az değer var. Bunları Rare edip öyle değerlenirebiliriz
######################


# 1-) Kategorik değişkenin azlık çokluk durumu
# 2-) Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi
# 3-) Rare encoder yazacağız.

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/application_train.csv")

df.head()

df["NAME_EDUCATION_TYPE"].value_counts()

# Secondary / secondary special    218391
# Higher education                  74863
# Incomplete higher                 10277
# Lower secondary                    3816
# Academic degree                     164   # bu diğerlerine göre çok az
# Name: NAME_EDUCATION_TYPE, dtype: int64

# Kategorik kolonlara ihtiyacımız var.

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

# Observations: 307511
# Variables: 122
# cat_cols: 54
# num_cols: 67
# cat_but_car: 1
# num_but_cat: 39

# Kategorik değişkenlerin sınıfları içinde kaç adet değer var?

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

# ##########################
#                       NAME_INCOME_TYPE      Ratio
# Working                         158774  51.631974
# Commercial associate             71617  23.289248
# Pensioner                        55362  18.003258
# State servant                    21703   7.057634
# Unemployed                          22   0.007154    # diğerlerine göre az
# Student                             18   0.005853    # diğerlerine göre az
# Businessman                         10   0.003252    # diğerlerine göre az
# Maternity leave                      5   0.001626    # diğerlerine göre az
# ##########################
#                                NAME_EDUCATION_TYPE      Ratio
# Secondary / secondary special               218391  71.018923
# Higher education                             74863  24.344820
# Incomplete higher                            10277   3.341994     # diğerlerine göre az
# Lower secondary                               3816   1.240931     # diğerlerine göre az
# Academic degree                                164   0.053331     # diğerlerine göre az
# ##########################


### Veri setinde Target değişkeni Krediyi ödeyebilmeyi ifade eder. 0'a yakın olanlar risk barındırmazken 1'e yakın olanlar ise risklidir.
# 2-) Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analizi

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# NAME_INCOME_TYPE
# Businessman             0.000000     # iş adamlarında risk yok
# Commercial associate    0.074843     # ticari faaliyetlerin riski az
# Maternity leave         0.400000
# Pensioner               0.053864
# State servant           0.057550
# Student                 0.000000     # öğrencilerin riski yok(çünkü hiç kullanmamış olabilirler)
# Unemployed              0.363636     # işsizlerin riski yüksek.
# Working                 0.095885

## Bu veri setini kabaca böyle yorumladık.

df.groupby("NAME_INCOME_TYPE")["TARGET"].count()

# NAME_INCOME_TYPE
# Businessman                 10
# Commercial associate     71617
# Maternity leave              5
# Pensioner                55362
# State servant            21703
# Student                     18
# Unemployed                  22
# Working                 158774
# Name: TARGET, dtype: int64


## Rare analizi yapan Bu işlemleri fonk. haline getireceğiz.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

# İşlte düm kat. değişkenler için toplam değer sayısı, tüm dfteki değerlere oranı ve hedef değişkendeki ortalaması
# burada bazı örnek değerler verilmiştir. Burada rare yapılacak sınıfları belirlemek için kullanabilriiz.

# nWEEKDAY_APPR_PROCESS_START : 7
#            COUNT     RATIO  TARGET_MEAN
# FRIDAY     50338  0.163695     0.081469
# MONDAY     50714  0.164918     0.077572
# SATURDAY   33852  0.110084     0.078873
# SUNDAY     16181  0.052619     0.079291
# THURSDAY   50591  0.164518     0.081003
# TUESDAY    53901  0.175282     0.083505
# WEDNESDAY  51934  0.168885     0.081604
# FONDKAPREMONT_MODE : 4
#                        COUNT     RATIO  TARGET_MEAN
# not specified           5687  0.018494     0.075435
# org spec account        5619  0.018273     0.058195
# reg oper account       73830  0.240089     0.069782
# reg oper spec account  12080  0.039283     0.065563
# HOUSETYPE_MODE : 3
#                    COUNT     RATIO  TARGET_MEAN
# block of flats    150503  0.489423     0.069434
# specific housing    1499  0.004875     0.101401
# terraced house      1212  0.003941     0.084983


# 3-) Rare encoder yazacağız.

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

# raito değeri 0.01 den küçük olanların hepsini birlerştirdi ve RARE yazdı,

# HOUSETYPE_MODE : 2
#                  COUNT     RATIO  TARGET_MEAN
# Rare              2711  0.008816     0.094061
# block of flats  150503  0.489423     0.069434
# WALLSMATERIAL_MODE : 5
#               COUNT     RATIO  TARGET_MEAN
# Block          9253  0.030090     0.070247
# Panel         66040  0.214757     0.063477
# Rare           5700  0.018536     0.068772
# Stone, brick  64815  0.210773     0.074057
# Wooden         5362  0.017437     0.096979


df["OCCUPATION_TYPE"].value_counts()

# Laborers                 55186
# Sales staff              32102
# Core staff               27570
# Managers                 21371
# Drivers                  18603
# High skill tech staff    11380
# Accountants               9813
# Medicine staff            8537
# Security staff            6721
# Cooking staff             5946
# Cleaning staff            4653
# Private service staff     2652
# Low-skill Laborers        2093
# Waiters/barmen staff      1348
# Secretaries               1305
# Realty agents              751
# HR staff                   563
# IT staff                   526
# Name: OCCUPATION_TYPE, dtype: int64

new_df["OCCUPATION_TYPE"].value_counts()

# aralarındaki farka yakından bakalım

# Laborers                 55186
# Sales staff              32102
# Core staff               27570
# Managers                 21371
# Drivers                  18603
# High skill tech staff    11380
# Accountants               9813
# Rare                      9238
# Medicine staff            8537
# Security staff            6721
# Cooking staff             5946
# Cleaning staff            4653


########################
# FEATURE SCALİNG (ÖZELLİK ÖLÇEKLENDİRME)
########################

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")
df.head()

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

ss = StandardScaler()
df["age_standart_saler"] = ss.fit_transform(df[["Age"]])

df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

# Robust daha cazip (aykırı değerler için)

# age_standart_saler  714.0  2.174187e-16    1.000701 -2.016979   -0.659542   -0.117049    0.571831    3.465126
# Age_robuts_scaler   714.0  9.505553e-02    0.812671 -1.542937   -0.440559    0.000000    0.559441    2.909091


###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

# age_standart_saler  714.0  2.174187e-16    1.000701 -2.016979   -0.659542   -0.117049    0.571831    3.465126
# Age_robuts_scaler   714.0  9.505553e-02    0.812671 -1.542937   -0.440559    0.000000    0.559441    2.909091
# Age_min_max_scaler  714.0  3.679206e-01    0.182540  0.000000    0.247612    0.346569    0.472229    1.000000


age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

# 3 ü de birbirinin aynısı. Sadece ölçekleri artık farklıdır.

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

df.head()



# Sonuç olarak şunları yaptık:


# 1-) Label Encoding (Binary-Büyüklük anlamı ifade edebilecek)

def label_encoding(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# 2-) One-Hot Encoding ( birden fazla değere sahip olan sınıflar için)

pd.get_dummies(df, columns=["Embarked", "Sex"], drop_first=True)

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if  10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols, drop_first=True)


# 3-) Rare Analizi - Kategorik değişkenlerin sınıfları içinde kaç değer var, nasıl dağılmış?

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col)

###

## Rare analizi yapan Bu işlemleri fonk. haline getireceğiz.

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

###


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)


# 3-) FEATURE SCALİNG (ÖZELLİK ÖLÇEKLENDİRME)

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

ss = StandardScaler()
df["age_standart_saler"] = ss.fit_transform(df[["Age"]])

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])

# Robust daha cazip (aykırı değerler için)

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])

# age_standart_saler  714.0  2.174187e-16    1.000701 -2.016979   -0.659542   -0.117049    0.571831    3.465126
# Age_robuts_scaler   714.0  9.505553e-02    0.812671 -1.542937   -0.440559    0.000000    0.559441    2.909091
# Age_min_max_scaler  714.0  3.679206e-01    0.182540  0.000000    0.247612    0.346569    0.472229    1.000000

# Sayısal değişkenlerin değerlerini görme

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)




