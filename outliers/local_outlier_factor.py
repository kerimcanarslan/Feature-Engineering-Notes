##############################
# ÇOK DEĞİŞKENLİ AYKIRI DEĞER ANALİZİ (LOF)
##############################

# 3 KERE EVLENMEK AYKIRI DEĞER OLMAYABİLİR.
# 17 YAŞINDA OLMA DURUMU DA AYKIRI OLMAYABİLİR.
# AMA 17 YAŞINDA 3 KERE EVLENMEK AYKIRI BİR DURUMDUR.

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score


df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in df.columns:
    print(col, check_outlier(df, col))    # carat True, depth True, table True, price True (hepsinde aykırı değer var)

low, up = outlier_thresholds(df, "carat")   # carat değişkeninin aykırı değerlei için alt ve üst limitleri

df[((df["carat"] < low) | (df["carat"] > up))].shape   # (1889, 7)  1889 adet aykırı değer var. Her kolonda durum böyle Çok fazla


##############
# LOF Yöntemi
##############

clf = LocalOutlierFactor(n_neighbors=20)
# Ön tanımlı değer 20'dir. Sektör bilgisine sahipsek 5-7-9-13-15-20-25-28-30 gibi değerlerden bazılarını
# deneyeyebiliriz. Aldığımız sonuçları yorumlayarak en doğrusunu seçebiliriz.
# burada sektörel bilgimiz olmadığı için ön tanımlı değer olan 20 yi seçiyoruz.

clf.fit_predict(df)   # df'imizi fit ettik. Ve aykırı değerler için LOF scorelar oluşacak.

df_scores = clf.negative_outlier_factor_

df_scores[0:5]  # ilk 5: array([-1.58352526, -1.59732899, -1.62278873, -1.33002541, -1.30712521])

# df_scores 1'e yakınlık durumuna göre aykırılık belirler. burada negatif değerlere bakıcaz
# -1 e ne kadar uzaksa o kadar aykırı olur.
# bir eşik değer belirleyip o değerin altında kalanları aykırı olarak nitelendireceğiz.
# bunu tespit edebilmek için dirsek yöntemi kullanacağız.

np.sort(df_scores)[0:5]   # array([-8.60430658, -8.20889984, -5.86084355, -4.98415175, -4.81502092])

# dirsek yöntemi

scores = pd.DataFrame(np.sort(df_scores))

scores.plot(stacked=True, xlim=[0, 10], style='.-')
plt.show()

# Grafikteki dirsek noktası (en sert değişim noktası) eşik değer olarak belirlenir.
# bu grafikte 3.indexten sonra birden aşağı hareket var. 3.index'i seçiyoruz

th = np.sort(df_scores)[3]   # -4.984151747711709 altında kalanlar aykırıdır diyeceğiz.

df[df_scores < th]  # 3 adet değer var.
df[df_scores < th].shape  # (3, 7)

# bunlar neden aykırı bilmiyorum, çünkü sektörel bilgim yok.

# Ağaç yöntemlerinde aykırılıklar sorun olmaz. Fakat yine de aykırılıklara dokunmak istiyorsak
# 0.99 - 0.01 tıraşlama yapılabilir.
# Burada 3 değer var zaten, silinse de büyük bir kayıp olmaz.

# Silmek için

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

# Tıraşlama tercih edilmelidir.














