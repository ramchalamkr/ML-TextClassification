import nltk
from nltk.corpus import stopwords
#from stemming.porter import stem
import pickle
from nltk.stem.porter import PorterStemmer
import re
import string
import pandas as pd
import numpy as np
stemmer = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

#execute this for the final test data prediction. After executing MultinomialNB.py and BernouliNB.py
def splitText(text):
    word_list = re.findall(r"[\w']+|[\*\+\-\/\<\>\=\~\%\#\^]", text) #split text and includes only words,numbers and specific symbols.
    #word_list = re.findall(r"[\w']+", text) 
    #word_list = re.findall(r"[\w']+|[\S]", text) #split text by spaces and all symbols.
    word_list = [re.sub(r"[\*\+\-\/\<\>\=\~\%\#\^]", 'symxyzabcd',s) for s in word_list] #bundle every symbol with a common name
    word_list = [re.sub(r"\w*[\d]+\w*", 'Numxyzabcd',s) for s in word_list] #bundle every number with a common name which will lump everything together during vectorisation
    #word_list = re.findall(r"[\w']+|[\*\+\-\/\+\$\^\>\<]", text) #split text by spaces and all symbols.
    return word_list

def removeStopWords(word_list):
    """ Removes stop words from text """
        
    cachedStopWords = set(stopwords.words("english"))    
    filtered_words = [w for w in word_list if not w in cachedStopWords]    
    return filtered_words

def stemWords(word_list):
    stemmedWords = [stemmer.stem(w) for w in word_list]
    return stemmedWords

def preProcessData(abstract):
    #preprocessing: stopword removal and stemming       
    word_list = splitText(abstract)
    word_list = removeStopWords(word_list)
    word_list = stemWords(word_list)
    return ' '.join(word_list)

#load estimators (vectoriser, label encoder, grid search)
bestEstimator = pickle.load(open('fittedGridSearchEstimator.pkl',"rb"))
vectoriser = pickle.load(open('vectoriser.pkl',"rb"))
le = pickle.load(open('labelEncoder.pkl',"rb"))

#load test data
df = pd.read_csv('test_in.csv')
test_abstracts = df['abstract'].as_matrix()

#preprocess
processedTestAbstracts = [preProcessData(a) for a in test_abstracts]

#vectorise
x_test = vectoriser.transform(processedTestAbstracts)

#execute learner estimator (grid search)
y_test = bestEstimator.predict(x_test)

#save in format
y_test_names = le.inverse_transform(y_test)
mydict = {'id': np.arange(0,len(y_test)),'category':y_test_names}
y_test_df = pd.DataFrame(mydict, columns=['id', 'category'])
y_test_df.to_csv('predictions_bin_chi2_NB.csv', index=False)
