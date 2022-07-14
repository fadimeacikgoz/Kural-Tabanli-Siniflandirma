#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları
# (persona)  oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek
# müşterilerin şirkete # ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı


################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C



import pandas as pd
import numpy as np
import seaborn as sns


# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("/Users/fadimeacikgoz/PycharmProjects/Python/datasets/persona.csv")
df.describe().T
df.head()
#######################################
#  PRICE   SOURCE   SEX COUNTRY  AGE
#0     39  android  male     bra   17
#1     39  android  male     bra   17
#2     49  android  male     bra   17
#3     29  android  male     tur   17
#4     49  android  male     tur   17
#######################################


# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"]
df["SOURCE"].unique()
# Out[104]: array(['android', 'ios'], dtype=object)

df["SOURCE"].nunique()
# Out[105]: 2


#Soru 3: Kaç unique PRICE vardır?
df["PRICE"]
df["PRICE"].nunique()
df["PRICE"].unique()
# Out[106]: 6


#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()
#29    1305
#39    1260
#49    1031
#19     992
#59     212
#9      200
#Name: PRICE, dtype: int64



#Soru 5: Hangi ülkeden kaçar tane satış(PRICE)  olmuş?
#ilk :kesisimlerde ne görmek istiyoruz  , ikinci : indexte satırda görmek istedigimiz , ücüncü : sutunda görmek istedigimiz !!!!!!!!!!
#Groupby ile yapalımç
df.groupby(["COUNTRY"]).agg({"PRICE": ["count"]})

#        PRICE
#        count
#COUNTRY
#bra      1496
#can       230
#deu       455
#fra       303
#tur       451
#usa      2065


#Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df("COUNTRY")["PRICE"].sum()
df.groupby(["COUNTRY"]).agg({"PRICE": ["sum"]})

#         PRICE
#            sum
# COUNTRY
# bra      51354
# can       7730
# deu      15485
# fra      10177
# tur      15689
# usa      70225


#Soru 7: SOURCE türlerine göre satış sayıları nedir?
df.groupby(["SOURCE"]).agg({"PRICE": ["count"]})
#        PRICE
#         count
# SOURCE
# android  2974
# ios      2026


# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby(["COUNTRY"]).agg({"PRICE": ["mean"]})
#            PRICE
#              mean
#COUNTRY
#bra      34.327540
#can      33.608696
#deu      34.032967
#fra      33.587459
#tur      34.787140
#usa      34.007264


#  Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(["SOURCE"]).agg({"PRICE": ["mean"]})
#             PRICE
#              mean
#SOURCE
#android  34.174849
#ios      34.069102

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY", "SOURCE"]).agg({"PRICE": ["mean"]})

#                      PRICE
#                       mean
# COUNTRY SOURCE
# bra     android  34.387029
#         ios      34.222222
# can     android  33.330709
#         ios      33.951456
# deu     android  33.869888
#         ios      34.268817
# fra     android  34.312500
#         ios      32.776224
# tur     android  36.229437
#         ios      33.272727
# usa     android  33.760357
#         ios      34.37170



########GOREV 2 #############
#COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": ["mean"]})

#                                 PRICE
#                                  mean
# COUNTRY SOURCE  SEX    AGE
# bra     android female 15   38.714286
#                        16   35.944444
#                        17   35.666667
#                        18   32.255814
#                        19   35.206897
#                                ...
# usa     ios     male   42   30.250000
#                        50   39.000000
#                        53   34.000000
#                        55   29.000000
#                        59   46.500000



#Pivot_table ile
#ilk :kesisimlerde ne görmek istiyoruz  , ikinci : indexte satırda görmek istedigimiz , ücüncü : sutunda görmek istedigimiz !!!!!!!!!!
table = pd.pivot_table(df, values='PRICE', index=['COUNTRY'],
                    columns=['SOURCE', 'SEX', 'AGE'], aggfunc=np.mean)


# Görev 3: Çıktıyı PRICE’a göre sıralayınız.
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE’a göre uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE": "mean"})
agg_df.sort_values("PRICE", ascending=False , inplace=True)

#                             PRICE
# COUNTRY SOURCE  SEX    AGE
# bra     android male   46    59.0
# usa     android male   36    59.0
# fra     android female 24    59.0
# usa     ios     male   32    54.0
# deu     android female 36    49.0
#                            ...
# usa     ios     female 38    19.0
#                        30    19.0
# can     android female 27    19.0
# fra     android male   18    19.0
# deu     android male   26     9.0



# Gorev 4 :Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir. Bu isimleri değişken isimlerine çeviriniz.
agg_df.reset_index(inplace=True)
agg_df
#     COUNTRY   SOURCE     SEX  AGE  PRICE
# 0       bra  android    male   46   59.0
# 1       usa  android    male   36   59.0
# 2       fra  android  female   24   59.0
# 3       usa      ios    male   32   54.0
# 4       deu  android  female   36   49.0
# ..      ...      ...     ...  ...    ...
# 343     usa      ios  female   38   19.0
# 344     usa      ios  female   30   19.0
# 345     can  android  female   27   19.0
# 346     fra  android    male   18   19.0
# 347     deu  android    male   26    9.0



#Görev 5: Age değişkenini kategorik değişkene çeviriniz ve agg_df’e ekleyiniz.
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici şekilde oluşturunuz.
# Örneğin: ‘0_18', ‘19_23', '24_30', '31_40', '41_7'

agg_df["Age_Cat"] = pd.cut(x=agg_df["AGE"], bins=[0, 18, 23, 30, 40, 70], labels=['0_18', '19_23', '24_30', '31_40', '41_70'])
#0, 18, 23, 30, 40, 70
agg_df.head()
#   COUNTRY   SOURCE     SEX  AGE  PRICE Age_Cat
# 0     bra  android    male   46   59.0   41_70
# 1     usa  android    male   36   59.0   31_40
# 2     fra  android  female   24   59.0   24_30
# 3     usa      ios    male   32   54.0   31_40
# 4     deu  android  female   36   49.0   31_40




# Gorev 6 =
#Yeni seviye tabanlı müşterileri (persona) tanımlayınız ve veri setine değişken olarak ekleyiniz.
# Yeni eklenecek değişkenin adı: customers_level_based
# Önceki soruda elde edeceğiniz çıktıdaki gözlemleri bir araya getirerek customers_level_based değişkenini oluşturmanız gerekmektedir

agg_df["customers_level_based"] = ["_".join(col) for col in agg_df.drop(["AGE", "PRICE"], axis=1).values]
agg_df["customers_level_based"] = ["_".join(col) for col in agg_df[["COUNTRY", "SOURCE", "SEX", "Age_Cat"]].values]
#for i in agg_df.columns : if agg_df[i].dtypes not in ["int64", "float64"]:

type(agg_df[["COUNTRY"]].values)

agg_df.head()
#   COUNTRY   SOURCE     SEX  AGE  PRICE Age_Cat     customers_level_based
# 0     bra  android    male   46   59.0   41_70    bra_android_male_41_70
# 1     usa  android    male   36   59.0   31_40    usa_android_male_31_40
# 2     fra  android  female   24   59.0   24_30  fra_android_female_24_30
# 3     usa      ios    male   32   54.0   31_40        usa_ios_male_31_40
# 4     deu  android  female   36   49.0   31_40  deu_android_female_31_40




# Gorev 7 = Yeni müşterileri (personaları) segmentlere ayırınız.
# • Yeni müşterileri (Örnek: USA_ANDROID_MALE_0_18) PRICE’a göre 4 segmente ayırınız.
# • Segmentleri SEGMENT isimlendirmesi ile değişken olarak agg_df’e ekleyiniz.
# • Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).
# cut fonksiyonu  = say ısal degiskeni kategorik degiskene dönüştürmeye yarar bilinen bir bir sayısal degiskense cut fonksyonu kullanılır
# qcut fonksiyonu = dönüştürülecek sayısal degiskenin degerleri belli degilse qcut kullanılır


agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df["SEGMENT"].head()
agg_df.head()
agg_df["SEGMENT"]

# 0      A
# 1      A
# 2      A
# 3      A
# 4      A
#       ..
# 343    D
# 344    D
# 345    D
# 346    D
# 347    D
# Name: SEGMENT, Length: 348, dtype: category
# Categories (4, object): ['D' < 'C' < 'B' < 'A'



# Segmentleri betimleyiniz (Segmentlere göre group by yapıp price mean, max, sum’larını alınız).

df.groupby(agg_df["SEGMENT"]).agg({"PRICE": ["mean", "max", "sum"]})
#              PRICE
#               mean max   sum
# SEGMENT
# D        34.977011  59  3043
# C        33.000000  59  3135
# B        34.185185  59  2769
# A        35.117647  59  2985

#Görev 8: Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini tahmin ediniz.
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?


new_user1 = "tur_android_female_31_40"
agg_df[agg_df["customers_level_based"] == new_user1]

#    COUNTRY   SOURCE     SEX  AGE      PRICE Age_Cat     customers_level_based SEGMENT
# 18     tur  android  female   32  43.000000   31_40  tur_android_female_31_40       A
# 35     tur  android  female   31  40.666667   31_40  tur_android_female_31_40       A


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user2 ="fra_ios_female_31_40"
agg_df[agg_df["customers_level_based"] == new_user2]

#   COUNTRY SOURCE     SEX  AGE      PRICE Age_Cat customers_level_based SEGMENT
# 208     fra    ios  female   40  33.000000   31_40  fra_ios_female_31_40       C
# 221     fra    ios  female   31  32.636364   31_40  fra_ios_female_31_40       C