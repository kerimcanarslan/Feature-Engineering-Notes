

import pandas as pd

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")
df.head()


df.describe().T


#### Outlier Thresholds Fonk.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low_limit, up_limit = outlier_thresholds(df, "Age")


### Aykırı değer var mı sorusunu soralım

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Fare")


### Sinsirellaları yakalama Fonk
### GRAB COL NAME

# Sayısal değişken (int,float) olarak görünen fakat sayısal olmayan değişkenler scriptte bize sorun çıkarabilir
# Değişken eğer 10'dan az eşşsiz değer barındırıyorsa bu muhtemelen kategoriktir.(Proje özelinde yoruma açıktır.)
# Dolayısıyla; Sayısal gibi görünen ama kategorik olanaları ayırmam lazım.
# Kategorik olarak görünüp aslında sayısal değişken olanları da ayırıp öyle devam etmeliyim

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

cat_cols
num_cols
cat_but_car

num_cols = [col for col in num_cols if col not in "PassengerId"]

# Sayısal değişkenlerde aykırı değer olup olmadığını gösteren bir script yazalım

for col in num_cols:
    print(col, check_outlier(df, col))


# Bunu daha büyük bir df te deneyelim

dff = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/application_train.csv")
dff.head()

cat_cols, num_cols, cat_but_car = grab_col_name(dff)

# SK_ID_CURR kullanıcı ID leri. İstersek çıkarabiliriz.

# bu veri seti için aykırı değer var mı yok mu diye soralım

for col in num_cols:
    print(col, check_outlier(dff, col))


### Aykırı değerlerin kendilerine ve index bilgilerine erişmek

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


# titanik içinde Age değişkeninde aykırı değerler varsa bunların indexini alalım

outlier_index = grab_outliers(df, "Age", index=True)
outlier_index   # Int64Index([33, 54, 96, 116, 280, 456, 493, 630, 672, 745, 851], dtype='int64')



#### Aykırı değerleri tesipt ettik evet ama şimdi bunu çözmek lazım.
# silebiliriz, güncelleyebiliriz, veya göz ardı edebiliriz
# Bu projeden projeye göre değişiklik gösterir


#########
## SİLME
#########

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]
num_cols

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]  # 891 - 775 = 116. Toplamda 116 aykırı değer sil,ndi ve new_df artık aykırı değer içermiyor.


#########
## BASKILAMA
#########

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit    # alt limitten daha aşağıda olanları alt limite eşitle.
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit      # üst limitten daha yukarda olanaları üst limite eşitle

# şimdi sıfırdan göstereyim.

df = pd.read_csv("Feature Engineering/feature_engineering/VBO_format/datasets/titanic.csv")

df.shape  #(891, 12)

cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# Outlier var mı soralım

for col in num_cols:
    print(col, check_outlier(df, col))
# Age True
# Fare True

# Şimdi aykıırı değerleri sınırlara eşitelyelim

for col in num_cols:
    replace_with_thresholds(df, col)

# şimdi tekrar soralım aykırı değer var mı diye

for col in num_cols:
    print(col, check_outlier(df, col))
# Age False
# Fare False
# Artık aykırı değer kalmadı. hepsi limitlere eşitlendi.








# Sonuç olarak 6 şey yaptık;

# 1-) Aykırı değerler için limit belirledik.

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low_limit, up_limit = outlier_thresholds(df, "Age")

# 2-) Kolonlar içinde bu limitleri aşan aykırı değerler var mı diye sorduk.

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

# 3-) Sinsirellaları tespit ettik, ve onlar üzerinde çalıştık.

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

# 4-) Aykırı değerlerin index bilgilerine ulaştık.

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

outlier_index = grab_outliers(df, "Age", index=True)

# 5-) Aykırı değerleri silmek gerekiyorsa.

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_name(df)

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

# 6-) Aykırı değerleri baskılamamız gerekiyorsa.

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit    # alt limitten daha aşağıda olanları alt limite eşitle.
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit      # üst limitten daha yukarda olanaları üst limite eşitle

replace_with_thresholds(df, "Age") 

















