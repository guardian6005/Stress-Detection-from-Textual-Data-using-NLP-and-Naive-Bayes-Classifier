# -*- coding: utf-8 -*-
"""Stress_analysis_Code.ipynb"""


import numpy as np
import pandas as pd

fil=pd.read_csv('stress.csv')
fil.head()

fil.describe()  #This provides statistics for the respective columns which contains int only

fil.isnull().sum()

"""import nltk: This line imports the Natural Language Toolkit (NLTK) library, which is a popular library for working with human language data.

from nltk.corpus import stopwords: This line imports the NLTK's stopwords corpus, which is a collection of commonly used words in a language that are usually removed from text when doing text analysis.
"""

import nltk
from nltk.corpus import stopwords

"""re is a built-in Python module that provides support for regular expressions. Regular expressions are a powerful and flexible way to search, match, and manipulate text strings.

The re module allows you to search for specific patterns within a string, and then use that pattern to match and manipulate text. It provides a variety of functions and methods for working with regular expressions, including search(), match(), findall(), sub(), and more.

import string: This line imports Python's built-in string module, which contains a collection of commonly used string operations.
"""

import re
import string

nltk.download('stopwords')                  #stopwords is a collection of common words in a language that are typically removed from text when doing natural language processing (NLP) tasks
stemmer=nltk.SnowballStemmer('english')     #stemming algorithm which stems words like [playing---> play]
stopword=set(stopwords.words('english'))    #making set of this stopwords

# Cleaning our text 

def clean(text):
    text = str(text) . lower()                                              #returns a string where all characters are lower case. Symbols and Numbers are ignored.
    text = re. sub('\[.*?\]',' ',text)                                      #substring and returns a string with replaced values.
    text = re. sub('https?://\S+/www\. \S+', ' ', text)                     #whitespace char with pattern
    text = re. sub('<. *?>+', ' ', text)                                    #special char enclosed in square brackets
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)    #eliminate punctuation from string
    text = re. sub(' \n',' ', text)                                         #removing newline paranthesis
    text = re. sub(' \w*\d\w*' ,' ', text)                                 #word character ASCII punctuation
    text = [word for word in text. split(' ') if word not in stopword]     #removing stopwords
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]            #remove morphological affixes from words
    text = " ". join(text)
    return text

# Now use this created function 

fil['text']=fil['text'].apply(clean)

"""Here `apply()` is a method available in Pandas that allows you to apply a function along an axis of a DataFrame or a Series. The function is applied to each element of the DataFrame or Series, and the result is returned as a new DataFrame or Series."""

# Simple visualization
import matplotlib. pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
text = " ".join(i for i in fil.text)
stopwords = set (STOPWORDS)
wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)
plt. figure(figsize=(10, 10) )
plt. imshow(wordcloud )
plt. axis("off")
plt. show()

"""CountVectorizer is a tool that transforms text data into numerical features by counting the occurrences of words, which can be used for machine learning models."""

from sklearn. feature_extraction. text import CountVectorizer
from sklearn. model_selection import train_test_split

x = np.array (fil["text"])
y = np.array (fil["label"])

cv = CountVectorizer ()
X = cv. fit_transform(x)
print(x.size)  #this is small x
print(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)

""" the first row (0) in the output shows that the word with index 7405 appears once in the first document, the word with index 3278 appears once, the word with index 9454 appears once, and so on. This means that the first document contains these words only once each."""

from sklearn.naive_bayes import BernoulliNB
model=BernoulliNB()
model.fit(xtrain,ytrain)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculation of Mean Squared Error (MSE)
y_train_predict=model.predict(xtrain)
trainingMSE=mean_squared_error(ytrain,y_train_predict)
print("MSE is :", trainingMSE)

#calculation of R^2
r2_train=r2_score(ytrain,y_train_predict)
print("R^2 is :", r2_train)

user=input("Enter the text : \n")
data=cv.transform([user]).toarray()
output=model.predict(data)
if output==1:
  print('The author of text is in stress')
else:
  print('no stress')
