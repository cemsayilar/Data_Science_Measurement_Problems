# MEASUREMENT PROBLEMS
# What are the reasons that make people buy? First of all we talk about 'social proof'. Especially last 10 year,
# the Wisdom of Crowds get more important. If Mehmet selects two mouses and one of them have 4 other one
# has 5 stars. But if 4 star one have rated by higher amount of people, most of us will choose low star but high number
# of rating.
# Making these social proof reliable, companys should choose and develop these carrefuly;
# Product Rating point calc.
# Product listing order
# Product commend listing order
# Product pages (main and selling) visual design.
# Testing feature changes on website or app to make better updates.
# Testing probable action and reactions. (If I take action like that, what will be the customers respond to that?)
# Methods for measureing these metrics;
# Rating Products
# Sorting Products
# Sorting Reviews
# AB Testing
# Lets begin.

# Rating Products
# Average
import pandas as pd
import numpy as np
import math
import scipy.stats as st
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats._stats_py import ttest_ind
import matplotlib as mt
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.width', 500)

df_ = pd.read_csv('/Users/buraksayilar/Desktop/measurement_problems/datasets/course_reviews.csv')
df = df_.copy()
df.head()
df.shape
df['Questions Asked'].value_counts()
df.groupby('Questions Asked').agg({'Questions Asked': 'count',
                                   'Rating':'mean'})
## Average
# If total rating is just calculeted by the mean, then ratings can not be accurate
# for the recent time. Both possitive and negative trends.
# So, Time-Based Weighted Average can be the answer.
#Time-Based Weighted Average

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
current_date = pd.to_datetime('2021-02-10 0:0:0')

df['days'] = (current_date - df['Timestamp']).dt.days
df[df['days'] < 30].count()
df.loc[df['days'] <=30, 'Rating'].mean()
df.loc[(df['days'] > 30) & (df['days'] <= 90) , 'Rating'].mean()
df.loc[(df['days'] > 90) & (df['days'] <= 180) , 'Rating'].mean()
# This means, people are more setisfied recently.
# We can make a little change to make time based weighted rating.
df.loc[df['days'] <=30, 'Rating'].mean() * 28/100 + \
df.loc[(df['days'] > 30) & (df['days'] <= 90) , 'Rating'].mean() * 26/100 + \
df.loc[(df['days'] > 90) & (df['days'] <= 180) , 'Rating'].mean() * 24/100 + \
df.loc[(df['days'] > 180), 'Rating'].mean() * 22/100
def time_based_we_average(dataframe,w1=28,w2=26,w3=24,w4=22):
    return dataframe.loc[dataframe['days'] <= 30, 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(dataframe['days'] > 30) & (dataframe['days'] <= 90), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(dataframe['days'] > 90) & (dataframe['days'] <= 180), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(dataframe['days'] > 180), 'Rating'].mean() * w4 / 100


# When distributeing the weights, more I gave, less weight I can give. And summary
# of the weights should be equal 100 end of the day.
# Third or even forth values after the comma can be fatal in e-commerce. Keep that
# in mind.
# I talked about the system that calculates the rating. What about the users that
# gives the ratings? Are they all equal for the system?

## User-Based Weighted Average
# Lets think about an online course. Is it fair that both rating comes from two diffrent
# customer has equal weights? If their percentege of comlition is diffrent?
# Data Analyst question...
# So we can create user_weight
df.groupby('Progress').agg({'Rating':'mean'})
# I can use df.loc like before.
def user_based_we_average(dataframe,w1=28,w2=26,w3=24,w4=22):
    return dataframe.loc[(dataframe['Progress'] <= 10), 'Rating'].mean() * w1 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 10) & (dataframe['Progress'] <= 45), 'Rating'].mean() * w2 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 45) & (dataframe['Progress'] <= 75), 'Rating'].mean() * w3 / 100 + \
           dataframe.loc[(dataframe['Progress'] > 75), 'Rating'].mean() * w4 / 100
## Weighted Rating
def course_we_rating(dataframe, time_w=50, user_w=50):
    return time_based_we_average(dataframe) * time_w/100 + user_based_we_average(dataframe) * user_w/100
# So I created a function that calculates rating with respect to users interest to course and rating time.
# Both have the same weight on the rating by default. It can be change.
course_we_rating(df)
# For example, if user commitmant is more important to me, then I can change its weight.
course_we_rating(df, time_w=40, user_w=60)

## Sorting Products
# For example there is a job post and each candidate has a score of three; GPA, Language_exam, interview_score.
df_ = pd.read_csv('/Users/buraksayilar/Desktop/measurement_problems/datasets/product_sorting.csv')
df = df_.copy()
df.head()
# There is a problem. Even just looking at the ratings and purchase-comment, these 2 variable is not effective
# as they should be.
## Sorting with Rating Comment and Purchase
df = df.rename(columns = {'commment_count':'comment_count'})
df['purchase_count_scaled'] = MinMaxScaler(feature_range= (1,5)).fit(df[['purchase_count']]).transform(df[['purchase_count']])
df['comment_count_scaled'] = MinMaxScaler(feature_range= (1,5)).fit(df[['comment_count']]).transform(df[['comment_count']])
def weighted_sorting_score(dataframe,w1=28,w2=26,w3=24):
    return (dataframe['comment_count_scaled'] * w1 /100 +
            dataframe['purchase_count_scaled'] * w2 /100 +
            dataframe['rating'] * w3 /100)


##Bayesian Average Rating Score
# Sorting products 5 star rated
# OR
# Sorting products according to distribution of 5 star rating
# This time I will use distribution of _points for calculating a probabilistic average.
# Puan da????l??mlar?? ??zerinden a????rl??kl?? bir ??ekilde olas??l??klsal ortalama hesab?? yapar.
import math
import scipy.stats as st
# Rationality is not about knowing the facts, its about recognizing which facts are relevant.
# P(H|E) hypothesis holds given the evidence is true. As in, we are restrecting our view only the possibilities where
# the evidence holds.
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0: # Points
        return 0
    K = len(n)
    z = st.norm.ppf(1-(1-confidence)/2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N+K+1))
    return score
for col in [col for col in df.columns if 'point' in col]:
    bayesian_average_rating(col, confidence=0.95)
df['bar_score'] = df.apply(lambda x: bayesian_average_rating(x[['1_point','2_point','3_point','4_point','5_point']]), axis=1)
df.sort_values('bar_score', ascending=False).head()
# There is a problem with this rating however. If I look close enough, points with a low frequency of low points, can
# have a higher rank. This is a problem :)
# I can calculate statistically powerful ratings but.. more votes should more thrustworthy right?

## Hybrid Sorting
# BAR Score + other factors.
# Rating products
# Time based weighted average
# User based weighted avereage
# Weighted rating
# Bayesian Average Rating Score (from 5 star rankings)
## Bayesian average rating if its used for scoreing, it can cut the ratings. Its a probabilistic method. So ratings
## seam lower than it is.
# Sorting Products
# Sorting by Rating
# Sorting by Comment Count or Purchase Count
# Sorting by Rating, Comment and Purchase
# Sorting by Bayesian Average Rating Score (Sorting Products with 5 star rated)
# Hybrit Sorting: BAR Score + others
def hybrid_sorting_score(dataframe, bar_w = 60, wss_w=30):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[['1_point',
                                                                     '2_point',
                                                                     '3_point',
                                                                     '4_point',
                                                                     '5_point']]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w/100 + wss_score*wss_w/100
# So that I mergeD probablistic methods outcome and my datasets valuable data such as comment and purchase counts.
df['hybrid_sorting_score'] = hybrid_sorting_score(df)
# It can seem pretty much same as wss scores ranking. but, 3th course seem have a low rating but the thing is it managed
# to top 5 with high purchase and comments. On the other hand, there are courses that neighter have a high comment or
# purchase volume but somehow manage to top #20. Things become triccky because thats the main purpuse of the BAR rating.
# To keep relatively new or/and promessing courses up.
# BAR skor y??ntemi, hibrit bir s??ralamada, hen??z yeterli 'social proof' alamam???? ??r??nleri de yukar?? ta????r.

##Uygulama IMDB Movie Scoring & Sorting

df_ = pd.read_csv('/Users/buraksayilar/Desktop/measurement_problems/datasets/movies_metadata.csv')
df = df_.copy()
df.head()
df = df[['title','vote_average','vote_count']]
# Job is updateing the IMDB' top250 list. There is an alghoritm (skorlama) that used by the IMDB until 2015.
# Sorting with vote_average
df.sort_values('vote_average', ascending=False).head(20)
# Films that have only 1 vote counts has corrapting the set. So I can make a filter for it.
df[df['vote_count'] > 100].sort_values(by='vote_count', ascending=False).head()
df['vote_count'].describe()
# As it seems, filtering just vote counts is abviously not enough. So I can scale this vote counts with minmaxscaler
# to further invesstigation.
df['vote_count_scaled'] = MinMaxScaler(feature_range=(0,10)). \
                              fit(df[['vote_count']]). \
                              transform(df[['vote_count']])
# fit and transform methods can be used together with fitransform.
# And now i can multiply with vote average and vote count.
df['vote_average_count_score'] = df['vote_count_scaled'] * df['vote_average'] # Summary can be useful?
df.sort_values(by='vote_average_count_score', ascending=False).head(20)

## IMDB Weighted Rating
# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)
# r = vote average
# v = vote count
# M = minimum votes required to be listed in the top 250
# C = The mean vote across the whole report (currently 7.0)
# In this calculation, IMDB engineers give importance to 2 variables; first, threshold for get in the top250 second
# average of all votes.
# With this weigthing, the movies that voted above the minimum vote not effected from the average. But if the film
# is a 'newbie' than it takes adventage of the average weight.
# Ayn?? zamanda, puan d??????k olsa da yorum ya da bu durumda oy say??s??n??n daha ??nemli olmas?? sa??lan??yor.
M = 2500
C = df['vote_average'].mean()
def weighted_rating(r, v, M, C):
    return (v/(v+M) * r) + (M/(v+M) * C)
df['weighted_rating'] = weighted_rating(df['vote_average'], df['vote_count'], M, C)
## Bayesian Average Rating Score
df = pd.read_csv('/Users/buraksayilar/Desktop/measurement_problems/datasets/imdb_ratings.csv')
df = df.iloc[0:, 1:]
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(20)

## Sorting Reviews
# ??r??nleri s??ralarken ??unlar?? fark??nda olmal??y??m:
# i?? bilgisi gerektiren fakt??rler g??z ??n??ne al??nmal??
# Birden fazla fakt??r varsa, bunlar beraber g??z ??n??ne al??nmak i??in standartla??t??r??lmal?? ard??ndan etkilerinin
# fark?? varsa ya da olmal??ysa bu 'weight' yani a????rl??klar ile ifade edilmeli.
# ??statistiksel y??ntemleri i?? bilgisi i??eren y??ntemler ile harmanlamak her hal??karda iyi olacakt??r.
# Yorumlarda puan ??nemli de??ildir. Ama?? en iyi sosyal ispat?? sunmakt??r?
# User quality score son derce ??nemli olabilir.

# What shouldn't do?
# Up-Down Diffrence Score = (up ratings) - (down ratings)
def score_up_down_diff(up, down):
    return up - down
# Averaga Rating (up ratings) / (all rating)
def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

# Wilson Lower Bound Score
#With WLB, I can score interaction of two.
#Bernuolli- p parametresi ile g??ven aral?????? bulma. ??rneklem ??zerinden g??ven aral?????? hesaplan??r.
def wilson_lower_bound(up, down, confidence= 0.95):
    '''
    Wilson Lower Bound Score Calculation
    Bernoulli parameter is the lower limit of the p 'g??ven aral??????'.
    Score is used for sorting products
    Note: If score between 1-5, 1-3 signed negative 4-5 signes positive. But it has its own problems so bayesian
    average rating can be more useful.
    :param up: up count
    :param down: down count
    :param confidence: confidence
    :return: wilson_score: float
    '''
    n = up+down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)
wilson_lower_bound(2,0)
wilson_lower_bound(100,1)
# ??rne??in sonu??lar?? incelendi??inde, elimdeki ??rneklem i??in 'uprate'? oran??n??, %95 do??ruluk ve %5 yan??lma pay?? ile
# hangi aral??kta (g??ven aral??????) oldu??unu tahmin edebiliyor ve o aral??????n alt s??n??r??n?? al??yorum.
up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})
# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"],x["down"]), axis=1)
# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)
# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)
comments.sort_values("wilson_lower_bound", ascending=False)
comments.sort_values(by= 'wilson_lower_bound', ascending=False)



## AB Testing
# Compareing two diffrent groups or products.
# But first I should mention some basics about statistics.
# Sample = a good example that can represent a 'Population' well enough.

# Basic Statistics
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats._morestats import levene
## Sampling (??rnekleme)
populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()
np.random.seed(115)
orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean() # Results are pretty close.
# If I take diffrent samples from population and take the mean of their sum of means, I get more closer to mean of the
# population.
## Descriptive Statistic (Ke??if??i Veri Analizi, Betimsel ??statistik)
df = sns.load_dataset('tips')
df.describe().T
# If I sort values ascending, the quratal (??eyrek) values represents kind of a coordinates these values.
## Confidence Intervals
# Probabilistic interval of main parameter (maybe mean)
# Steps
# find n, mean and std.
# Decide the interval confidence; 95 or 99? (In most of the cases, it is 95) Then find Z value from the table.
# Final step, calculate the confidence interval. Now I have an interval for my mean value with 95% confidence.
# Intervals can calculate for proporiton and diffrence of proportions of events, diffrence of two mean values.
## Exercise
sms.DescrStatsW(df['total_bill']).tconfint_mean()

## Corelation
# Relationship between parameters. Power of this correlation and kind (positive of negative)
df.head()
df['total_bill'] = df['total_bill'] - df['tip']
df.plot.scatter('tip', 'total_bill')
# plt.show()

## Hypothesis Testing
# Grup kar????la??t??rmalar??nda temel ama?? olas?? farkl??l??klar??n ??ans eseri ortaya ????k??p ????kmad??????n?? g??stermeye ??al????makt??r.
## AB Testing
# Yayg??nca, iki ortalama de??er ya da oran k??yaslan??yor.

# H0 benim yokluk hipotezimdir. S??nayacak oldu??um durum burada yer al??r.
# H1 hipotezi ise 'alternatif' hipotezdir.
# ??rnek say??lar??n??n ve varyanslar??n??n durumuna g??re kullan??lan form??ller de??i??ir.
# ??lgili belli hipotez testlerini ger??ekle??tirdi??imde, sonucu (yani 'p' de??eri) 0.05 ten k??????k ise H0 ?? reddedece??im.
# Bunun yan??nda, kulland??????m bu testlerin baz?? temel kabulleri vard??r. ??rne??in ba????ms??z ??ki ??rneklem T Testinde bu kab??ller
# iki grubun da normal da????l??ml?? olmas??, di??eri ise gruplar??n varyanslar??n??n homojenli??idir.
# Yani;
# 1- Hipotezi kur
# 2- Varsay??mlar?? incele. (Gerekirse veri ??n i??leme ve ke??if??i veri analizi i??lemleri yapar??m.)
# 3- p value kontrol?? ile yorum yap.
# ??rne??in, bir ??irketin m????terilerine ??r??n tahmini i??in kulland?????? ML algoritmes??n?? de??i??tirmi?? olsun. Bu sefer ????renmek
# istedi??im hangisinin daha iyi ??al????t?????? de??il, hangisinin sonunda daha ??ok para kazand??????md??r. Att??????m??z ta?? ??rk??tt??????m??z
# ku??a de??di mi? Vahit Keskin
# H0 : Yeni ML gelirde bir farkl??l??k olu??turmad??. (Genel kan??, mant??kl?? ak??lc?? olan, ve benim savundu??um.)
# H1 : Yeni ML gelirde bir farkl??l??k olu??turdu.
# t nin hesaplanan de??eri t nin tablo de??erinden b??y??kse, H0 hipotezi reddedilir.



## AB Testing (Ba????ms??z ??ki ??rneklem Testi)
# STEPS, AGAIN
# 1- Hipotezleri kur
# 2- Varsay??mlar?? incele. (Gerekirse veri ??n i??leme ve ke??if??i veri analizi i??lemleri yapar??m.)
#      1- Normallik (Normal da????l??m) varsay??m??
#      2- Varyans Homojenli??i
# 3- Hipotezin Kurulmas??
#      1- Varsay??mlar sa??lan??yorsa ba????ms??z iki ??rneklem testi (Parametrik Test)
#      2- Varsay??mlar sa??lanm??yorsa Mannwhitneyu testi (Non-parametrik Test)
# 4- p-value de??erini yorumla.
# NOT
# Normallik sa??lan??yorsa direkt iki numara. Varyans homojenli??i sa??lanm??yorsa 1 numaraya arg??man girilir. Varyans homojenli??inin sa??lanmad?????? belirtilir.
# Normallik incelemesi ??ncesi ayk??r?? de??er incelemesi ve d??zeltmesi yapmak faydal?? olabilir.
# Yani, e??er A/B testim iki grubun ortalamas??na y??nelikse bu ad??mlar?? izliyorum.

# Exercise 1
# Sigara i??enler ile i??meyenlerin hesap ortalamalar?? aras??nda istat. olarak anlaml?? fark var m???
df.groupby('smoker').agg({'total_bill': ['mean', 'sum', 'count'],
                          'tip': ['mean', 'sum', 'count']})
#1 Hipotezi kur
# H0: M1=M2  # ??ki grubun hesap ortalamalar?? e??it.
# H1: M1!=M2 # ??ki grubun hesap ortalamalar?? e??it de??il.
#2 Varsay??m Kontrol??
#Normallik Kontrol??
#Bir de??i??kenin da????l??m??n??n standart/normal da????l??ma benzer olup olmad??????n??n hipotez testidir.
#     H0: Normal da????l??m varsay??m?? sa??lanmaktad??r. (??statistiki olarak iki da????l??m aras??nda anlam. fark YOKTUR demek)
test_stat, pvalue = shapiro(df.loc[df['smoker'] == 'Yes', 'total_bill'])
# Shapiro testi bir de??i??kenin da????l??m??n??n normal olup olmad??????n?? test eder.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#### Varsay??mlar??n hipotezleri ile as??l yapt??????m hipotezin ele al??n?????? farkl??. As??l hipotezde H0 de??erini reddetmek istiyorum.
#### ancak varsay??m hipotezlerinde H0 lar?? (asl??nda) kabul etmek, do??ru ????kmalar??n?? istiyorum ki daha d??zg??n bir sonuca ula??ay??m.



#Varsay??m Kontrol??
# H0: Varyanslar homojendir
# H1: Varyanslar homojen de??ildir.
test_stat, pvalue = levene(df.loc[df['smoker'] == 'Yes', 'total_bill'],
                           df.loc[df['smoker'] == 'No', 'total_bill'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# ??K?? VARSAYIM DA REDDED??LD??

# 3 H??POTEZ??N UYGULANMASI
# VARSAYIMLAR SA??LANMI?? G??B?? YAPIYORUM
# Varsay??mlar sa??lan??yorsa ba????ms??z iki ??rneklem t testi (parametrik test)
test_stat, pvalue = ttest_ind(df.loc[df['smoker'] == 'Yes', 'total_bill'],
                              df.loc[df['smoker'] == 'No', 'total_bill'],
                              equal_var=True)
# ttest_ind'i varyans e??itli??i yokken yap??l??rsa, Welch testi ger??ekle??tirir. Ne oldu??unu bilmiyorum.
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-Value 0.05 ten b??y??k geldi. Yani H0 hipotezi reddedilemez. Bu da sigara i??enler ve i??meyenlerin b??rakt??klar?? hesap
# aras??nda istatistiki olarak anlaml?? bir ili??ki yoktur demektir.

# ASLINDA ??K?? VARSAYIM DA SA??LANMADI. O HALDE NE YAPACA??IM? (Mannwhitneyu, non-parametrik bir ortalama k??yaslama testidir.)
test_stat, pvalue = mannwhitneyu(df.loc[df['smoker'] == 'Yes', 'total_bill'],
                                 df.loc[df['smoker'] == 'No', 'total_bill'])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

###### H??POTEZ TEST?? YAPARKEN 'H0' ?? REDDEDER YA DA KABUL EDER??Z. 'H1' i KABUL ETMEK D??YE B??R KARARA VARMAK M??MK??N
###### DE????LD??R.
###### ????MK?? 'H0' REDDED??LD??????NDE YA DA REDDED??LMED??????NDE, YAPACAK OLDU??UMUZ HATA M??KTARINI B??L??R??Z, 0.05.
###### ANCAK 'H1' i KABUL ETT??????M??ZDE YAPACAK OLDU??UMUZ HATAYI B??LMEY??Z.

# EXERCISE-2
# Titanic Kad??n ve Erkek Yolcular??n Ya?? Ortalamalar?? Aras??nda ??stat. Olarak Anlaml?? Fark Var m???
df = sns.load_dataset('titanic')

df.groupby('sex').agg({'age':'mean'})
# 1-Hipotezleri Kur
# H0: M1 = M2 (Kad??n ve erkek ya?? ortalamalar?? aras??nda istatistiksel olarak anlaml?? fark yoktur.
# H1: M1 != M2 (Kad??n ve erkek ya?? ortalamalar?? aras??nda istatistiksel olarak anlaml?? fark vard??r.

# 2-Varsay??mlar?? ??ncele
# Normallik Varsay??m??
# H0: Normal da????l??m varsay??m?? sa??lan??r
# H1: Sa??lanmaz
test_stat, pvalue = shapiro(df.loc[df['sex'] == 'female', 'age'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p de??eri 0.05 ten k??????k oldu??u i??in H0 reddedilir.
test_stat, pvalue = shapiro(df.loc[df['sex'] == 'male', 'age'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p de??eri 0.05 ten k??????k oldu??u i??in H0 reddedilir.
# Bu testler sa??lanmad?????? i??in asl??nda direkt non-parametrik testlere y??nelmeliyim. Ancak varyans homojenli??ini de
# test edece??im
# Varyans Homojenli??i
# H0: Varyanslar homojendir
# H1: Varyanslar homojen de??ildir.
test_stat, pvalue = levene(df.loc[df['sex'] == 'female', 'age'].dropna(),
                           df.loc[df['sex'] == 'male', 'age'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Varsay??mlar sa??lanmad?????? i??in non-parametrik testleri kullanaca????m.
# Non-parametrik iki ??rneklem kar????la??t??rma testi mannwithneyu kullan??lacak.
test_stat, pvalue = mannwhitneyu(df.loc[df['sex'] == 'female', 'age'].dropna(),
                                 df.loc[df['sex'] == 'male', 'age'].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# Exercise-3
# Diyabet olanlar ve olmayanlar??n ya??lar?? aras??nda istat. olarak anlaml?? fark var m??d??r?

# Wilson lower bound metodu bize bir 'oran' i??in g??ven aral?????? verir.
# Belli bir g??zergahta yap??lan g??zlemde, 100 ara??tan 40 ??n??n kaza yapt?????? g??r??lm????se,
# wilson lower bound metodu bize der ki; bu deney y??z kere tekrar edilse, kaza yapan ara?? oran??
# y??zde 33 ila y??zde 40 aras??nda olacakt??r gibi bir aral??k verebilir.