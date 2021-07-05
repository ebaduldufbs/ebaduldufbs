
import bs4
import inline as inline
import matplotlib
import pandas as pd
import sklearn
assert sklearn.__version__>="0.20"
import sklearn.impute
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import sys
assert sys.version_info >= (3, 5)
import os
import inline
#to plot pretty figures
#matplotlib inline
import matplotlib as mpl
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import tarfile
import urllib
import urllib.request
from pandas.plotting import scatter_matrix
from scipy.stats import randint
import joblib
import nltk
import stopwords
#install and download all nltk packages
#nltk.download('stopwords')to download stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
#Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd # our main data management package
import matplotlib.pyplot as plt # our main display package
import string # used for preprocessing
import re # used for preprocessing
import nltk # the Natural Language Toolkit, used for preprocessing
import numpy as np # used for managing NaNs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # used for preprocessing
from nltk.stem import WordNetLemmatizer # used for preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # our model
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


df = pd.read_pickle("C:/Users/Besitzer/Downloads/df_london.pkl")
housing = df
#print(housing.info())
#housing.hist(bins=50,figsize=(10,10))
#plt.show()
#print(housing.count())

#token_list = nltk.word_tokenize(housing['description'])
#Text data description preprocessing
# split into words by white space
housing_description= str(housing["description"])
words = housing_description.split()
#print(words[:100])

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
stripped = [w.translate(table) for w in words]
#print(stripped[:100])

# convert to lower case
words = [word.lower() for word in words]
#print(words[:100])
# split into sentences
from nltk import sent_tokenize
sentences = sent_tokenize(housing_description)
#print(sentences[0])

# split into words
from nltk.tokenize import word_tokenize
tokens = word_tokenize(housing_description)
#print(tokens[:100])

# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
#print(words[:100])

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
#print(words[:100])

# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
#print(stemmed[:100])


#Beg_of_Words
# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 3. Create the Bag-of-Words Model
bag_of_words = count_vectorizer.fit_transform(sentences)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_words.toarray(), columns = feature_names)

#print(pd.DataFrame(bag_of_words.toarray(), columns = feature_names))

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(sentences)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
pd.DataFrame(values.toarray(), columns = feature_names)
#print(pd.DataFrame(values.toarray(), columns = feature_names))
housing_description_t= pd.DataFrame(values.toarray(), columns = feature_names)

#text data feature_items preprocessing

# split into words by white space
housing_feature_items= str(housing["feature_items"])
#print(housing_feature_items)
wordsfi = housing_feature_items.split()
#print(wordsfi[:100])

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
strippedfi = [w.translate(table) for w in wordsfi]
#print(strippedfi[:100])

# convert to lower case
wordsfi = [word.lower() for word in wordsfi]
#print(wordsfi[:100])

# split into sentences
from nltk import sent_tokenize
sentencesfi = sent_tokenize(housing_feature_items)
#print(sentencesfi[0])

# split into words
from nltk.tokenize import word_tokenize
tokensfi = word_tokenize(housing_feature_items)
#print(tokensfi[:100])

# remove all tokens that are not alphabetic
wordsfi = [word for word in tokensfi if word.isalpha()]
#print(wordsfi[:100])

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordsfi = [w for w in wordsfi if not w in stop_words]
#print(wordsfi[:100])

# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmedfi = [porter.stem(word) for word in tokensfi]
#print(stemmedfi[:100])


#Beg_of_Words
# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 3. Create the Bag-of-Words Model
bag_of_wordsfi = count_vectorizer.fit_transform(sentencesfi)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_wordsfi.toarray(), columns = feature_names)

#print(pd.DataFrame(bag_of_wordsfi.toarray(), columns = feature_names))

#TF-IDF
#from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#tfidf_vectorizer = TfidfVectorizer()
valuesfi = tfidf_vectorizer.fit_transform(sentencesfi)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
pd.DataFrame(valuesfi.toarray(), columns = feature_names)
#print(pd.DataFrame(valuesfi.toarray(), columns = feature_names))
housing_feature_items_t=pd.DataFrame(valuesfi.toarray(), columns = feature_names)

#Text data url preprocessing
# split into words by white space
housing_url= str(housing["url"])
wordsurl = housing_url.split()
#print(wordsurl[:100])

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
strippedurl = [w.translate(table) for w in wordsurl]
#print(strippedurl[:100])

# convert to lower case
wordsurl = [word.lower() for word in wordsurl]
#print(wordsurl[:100])
# split into sentences
from nltk import sent_tokenize
sentencesurl = sent_tokenize(housing_url)
#print(sentencesurl[0])

# split into words
from nltk.tokenize import word_tokenize
tokensurl = word_tokenize(housing_url)
#print(tokensurl[:100])

# remove all tokens that are not alphabetic
wordsurl = [word for word in tokensurl if word.isalpha()]
#print(wordsurl[:100])

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordsurl = [w for w in wordsurl if not w in stop_words]
#print(wordsurl[:100])

# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmedurl = [porter.stem(word) for word in tokensurl]
#print(stemmedurl[:100])


#Beg_of_Words
# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 3. Create the Bag-of-Words Model
bag_of_wordsurl = count_vectorizer.fit_transform(sentencesurl)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_wordsurl.toarray(), columns = feature_names)

#print(pd.DataFrame(bag_of_wordsurl.toarray(), columns = feature_names))

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

tfidf_vectorizer = TfidfVectorizer()
valuesurl = tfidf_vectorizer.fit_transform(sentencesurl)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
pd.DataFrame(valuesurl.toarray(), columns = feature_names)
#print(pd.DataFrame(valuesurl.toarray(), columns = feature_names))
housing_url_t=pd.DataFrame(valuesurl.toarray(), columns = feature_names)


#Text data property_type preprocessing
# split into words by white space
housing_pt= str(housing["property_type"])
wordspt = housing_pt.split()
#print(wordspt[:100])

# remove punctuation from each word
import string
table = str.maketrans('', '', string.punctuation)
strippedpt = [w.translate(table) for w in wordspt]
#print(strippedpt[:100])

# convert to lower case
wordspt = [word.lower() for word in wordspt]
#print(wordspt[:100])

# split into sentences
from nltk import sent_tokenize
sentencespt = sent_tokenize(housing_pt)
#print(sentencespt[0])

# split into words
from nltk.tokenize import word_tokenize
tokenspt = word_tokenize(housing_pt)
#print(tokenspt[:100])

# remove all tokens that are not alphabetic
wordspt = [word for word in tokenspt if word.isalpha()]
#print(wordspt[:100])

# filter out stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
wordspt = [w for w in wordspt if not w in stop_words]
#print(wordspt[:100])

# stemming of words
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stemmedpt = [porter.stem(word) for word in tokenspt]
#print(stemmedpt[:100])


#Beg_of_Words
# Import the libraries we need
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Step 2. Design the Vocabulary
# The default token pattern removes tokens of a single character. That's why we don't have the "I" and "s" tokens in the output
count_vectorizer = CountVectorizer()

# Step 3. Create the Bag-of-Words Model
bag_of_wordspt = count_vectorizer.fit_transform(sentencesurl)

# Show the Bag-of-Words Model as a pandas DataFrame
feature_names = count_vectorizer.get_feature_names()
pd.DataFrame(bag_of_wordspt.toarray(), columns = feature_names)

#print(pd.DataFrame(bag_of_wordspt.toarray(), columns = feature_names))

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

tfidf_vectorizer = TfidfVectorizer()
valuespt = tfidf_vectorizer.fit_transform(sentencespt)

# Show the Model as a pandas DataFrame
feature_names = tfidf_vectorizer.get_feature_names()
pd.DataFrame(valuespt.toarray(), columns = feature_names)
#print(pd.DataFrame(valuespt.toarray(), columns = feature_names))
housing_pt_t=pd.DataFrame(valuespt.toarray(), columns = feature_names)

#print(housing_pt_t)
