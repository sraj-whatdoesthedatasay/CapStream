# import libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

#header
st.title('Agent help questionnaire')

# user inputs
question_text = st.text_input("How can we help you?")

#Read in the Answer Corpus
ans_file = pd.read_csv('./data/IntentAnswers.csv')
ans = ans_file.set_index('intent')

# function to preprocess user inputs for nlp model
def preprocess_nlp(question):
    input_list = []
    processed_question = question.replace('-', '')
    tokenizer = RegexpTokenizer('\w+|\$[\d.]+|S+')
    token = tokenizer.tokenize(processed_question.lower())
    lemmatizer = WordNetLemmatizer()
    lem_token = [lemmatizer.lemmatize(word) for word in token]
    #tokens_filtered= [word for word in lem_token if not word in stopwords.words('english')]
    joined_text = ' '.join(lem_token)
    input_list.append(joined_text)
    return input_list

# loading models
cs_model = pickle.load(open('./models/cs_model.p', 'rb'))

# processing inputs for nlp model
input_text = preprocess_nlp(question_text)
ip_series = pd.Series(input_text)
answer_nlp = cs_model.predict(input_text)
ans.loc[answer_nlp[0]][0]
