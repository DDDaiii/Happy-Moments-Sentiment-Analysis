
# coding: utf-8

# In[469]:

##################################INTRODUCTION########################################
# In this report, I would like to focus on gender. Since there might be some intersting correlation between gender and other aspects of happy moments we humanbeings have. In the following report, I would explore the following three parts:
# 1. Number of sentence and gender
# 2. Sentiment analysis of text with gender
# 3. Word frequency and gender

## IMPORT PAKAGES
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


# In[470]:

import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
import plotly.graph_objs as go
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter 
import csv
from textblob import TextBlob
import nltk
import string
import nltk.corpus
import nltk.stem.snowball
from nltk.corpus import wordnet
import matplotlib.mlab as mlab  


# In[471]:

## lOAD DATA
clean = pd.read_csv('https://raw.githubusercontent.com/rit-public/HappyDB/master/happydb/data/cleaned_hm.csv',error_bad_lines=False)


# In[472]:

demo = pd.read_csv('https://raw.githubusercontent.com/rit-public/HappyDB/master/happydb/data/demographic.csv',error_bad_lines=False)


# In[473]:

## PREPARE DATA

# Merge two datasets we need

clean = pd.merge(clean, demo, how='inner', on='wid')


# In[474]:

clean.head()


# In[475]:

## CLEAN DATA

#  Clean gender and only include female and male

clean = clean[clean['gender'] != 'o']
clean.groupby('gender').sum()

# We can see that male constitutes the majority of the total population of this survey, the number of male is almost 20000 higher than female, which might have some impact on our following research.


# In[476]:

## Number of sentence and gender: 

# Group the gender with the number of sentence of happy moment
num = clean.groupby(["num_sentence", "gender"]).size()
num = num.sort_values(ascending=False).to_frame()
num.columns = ['count']

num = num[0:15]
num.plot.bar()
plt.show()

# In this bar chart, we can see that people, no matter male or female, would prefer to use one sentence to express their feelings when they are happy; generally,they would not use exceed three sentences for expression.


# In[477]:

# Compare the number of sentence male and female use to express their happy moments

x = [1, 2, 3, 4, 5, 6 ,7]
y =  []
for i in x:
    y.append(num.loc[(i,'m')]['count'])
z =  []
for i in x:
    z.append(num.loc[(i,'f')]['count'])
total_width, n = 0.6, 2
width = total_width / n
plt.bar(x, y, width=width, label='male',fc = 'gold')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, z, width=width, label='female',tick_label = x,fc = 'slateblue')
plt.legend()
plt.show()

## After considering different in the total sample size mentioned above, we can see that males are still more prone to use one to two sentence to express their happy moments, most of females have the same preference, while there are still some exceptions that they would use more than three sentences to elaborate their happyniess, which is consistent with our common sense that men tend to be more use concise and use less sentence. However, another factor needs to be considered - the length of sentence. By incoporating which into the analysis, we can better understand the difference between man and women in their expressions.    


# In[479]:

## POPULATION LEVEL SENTIMENT ANALYSIS
male = clean[clean['gender'] == 'm']
female = clean[clean['gender'] == 'f']

pop = ['male', 'female']
poppop = [male, female]
for i in range(len(poppop)):
    with open('sentiment_'+pop[i]+'.csv', mode='w') as topic:
        writer = csv.writer(topic, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['wid', 'polarity', 'subjectivity'])
        for index, stats in poppop[i].iterrows():
            text = TextBlob(stats['cleaned_hm'])
            writer.writerow([stats['wid'], text.sentiment.polarity, text.sentiment.subjectivity])


# In[480]:

female = pd.read_csv('sentiment_female.csv', error_bad_lines=False)
male = pd.read_csv('sentiment_male.csv', error_bad_lines=False)
# married = pd.read_csv('sentiment_married.csv', error_bad_lines=False)
# old = pd.read_csv('sentiment_old.csv', error_bad_lines=False)
# single = pd.read_csv('sentiment_single.csv', error_bad_lines=False)
# young = pd.read_csv('sentiment_young.csv', error_bad_lines=False)


# In[481]:

female_filtered = female.groupby('wid').mean()
male_filtered = male.groupby('wid').mean()


# In[482]:

# Disribution of female polarity
mu = 0.235 
sigma = 0.17  
x = mu + sigma * np.random.randn(10000)  
num_bins = 50 
n, bins, patches = plt.hist(female_filtered['polarity'], num_bins, normed=1, facecolor='blue', alpha=0.5)  

# Best fit line  
y = mlab.normpdf(bins, mu, sigma)  
plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title('Histogram of female polarity')  
   
plt.subplots_adjust(left=0.15)  
plt.show()  


# In[483]:

# Disribution of male polarity  
mu = 0.18   
sigma = 0.145   
x = mu + sigma * np.random.randn(10000)  
  
num_bins = 50
n, bins, patches = plt.hist(male_filtered['polarity'], num_bins, normed=1, facecolor='blue', alpha=0.5)  
y = mlab.normpdf(bins, mu, sigma)  
plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title('Histogram of male polarity')  
  
# Tweak spacing to prevent clipping of ylabel  
plt.subplots_adjust(left=0.15)  
plt.show()  


# In[484]:

# Disribution of female subjectivity   
mu = 0.48 
sigma = 0.2
x = mu + sigma * np.random.randn(10000)  
  
num_bins = 50
# the histogram of the data  
n, bins, patches = plt.hist(female_filtered['subjectivity'], num_bins, normed=1, facecolor='blue', alpha=0.5)  
 
# add a 'best fit' line  
y = mlab.normpdf(bins, mu, sigma)  
plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title('Histogram of female subjectivity')  
  
# Tweak spacing to prevent clipping of ylabel  
plt.subplots_adjust(left=0.15)  
plt.show()  


# In[485]:

# Disribution of male subjectivity    
mu = 0.45 # mean of distribution  
sigma = 0.2 # standard deviation of distribution  
x = mu + sigma * np.random.randn(10000)  
  
num_bins = 50
# the histogram of the data  
n, bins, patches = plt.hist(male_filtered['subjectivity'], num_bins, normed=1, facecolor='blue', alpha=0.5)  
 
# add a 'best fit' line  
y = mlab.normpdf(bins, mu, sigma)  
plt.plot(bins, y, 'r--')  
plt.xlabel('Smarts')  
plt.ylabel('Probability')  
plt.title('Histogram of male subjectivity')  
  
# Tweak spacing to prevent clipping of ylabel  
plt.subplots_adjust(left=0.15)  
plt.show()  

# These above four charts demonstrates that male and female share a similiar distribution of subjectivity and polarity. Besides, the mean of distribution of subjectivity is lower than that of polarity.   


# In[ ]:

female = clean.loc[clean['gender'] =='f']


# In[ ]:

# WORDCLOUD

stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(string.punctuation)
stopwords.append('')

## Cited from https://www.kaggle.com/ydalat/happydb-what-100-000-happy-moments-are-telling-us/notebook
nolist = ['happy', 'day', 'got', 'went', 'today', 'made', 'one', 'two', 'time', 'last', 'first', 'going',
'getting', 'took', 'found', 'lot', 'really', 'saw', 'see', 'month', 'week', 'day', 'yesterday',
'year', 'ago', 'now', 'still', 'since', 'something', 'great', 'good', 'long', 'thing', 'toi', 'without',
'yesteri', '2s', 'toand', 'ing', 'got', 'came', 'could', 'happiness', 'new', 'able', 'finally', 'like',
'old', 'years', 'many', '2', 'get', 'taj', 'nice', 'top', 'back']


def text_process(original):
    nopunc = [word for word in original if word not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split()
    if ((word.lower() not in stopwords) & (word.lower() not in nolist))]

female['cleaned_hm'] = female['cleaned_hm'].apply(text_process) 

for i, row in female.iterrows():
    female['cleaned_hm'][i] = ' '.join(female['cleaned_hm'][i])
    
print(female.dtypes)
text = ' '.join(female['cleaned_hm'].tolist())
wordcloud = WordCloud(background_color="white", height=3000, width=5500).generate(text)
plt.figure( figsize=(30,10) )
plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:

male = clean.loc[clean['gender'] =='m']
male['cleaned_hm'] = male['cleaned_hm'].apply(text_process) 

        
for i, row in male.iterrows():
    male['cleaned_hm'][i] = ' '.join(male['cleaned_hm'][i])
    
text = ' '.join(male['cleaned_hm'].tolist())
wordcloud = WordCloud(background_color="white", height=3000, width=5500).generate(text)
plt.figure( figsize=(30,10) )
plt.imshow(wordcloud.recolor(colormap=plt.get_cmap('Set2')), interpolation='bilinear')
plt.axis("off")
plt.show()

# From these two plots, we can see that there are some similarities in topics which would make male and female happy, most of them  are related to family, work and friend.
# For female, the top 5 topics are work, husband, friends, son and daughter
# For male, the top 5 topics are friend, work, wife, girlfriend and family


# In[ ]:

## In conclusion, I personally find it quite interesting that males and females are actually share many similarities in number of sentence to express happiness, subjectivity and polarity and also the topics which would make them happy.  


# In[ ]:

## CITATION: 
+ [Plot https://blog.csdn.net/qq_29721419/article/details/71638912]
+ [Plot https://blog.csdn.net/jenyzhang/article/details/52047557]
+ [wordcloud https://www.kaggle.com/ydalat/happydb-what-100-000-happy-moments-are-telling-us/notebook]

