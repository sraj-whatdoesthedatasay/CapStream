# import libraries
# Standard imports
import pandas as pd
import numpy as np
import re
import pickle
import string
import json
import streamlit as st
import nltk
nltk.download('wordnet')

# Visualization library
#import seaborn as sns
#import matplotlib.pyplot as plt

# NLP library
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.backend import manual_variable_initialization

manual_variable_initialization(True)

#header
st.title('Multiple NLP Algos')
st.write('Algos help with: Chatbot to answer card questions, Spanish translation, completing sentences, deducting whether post sounds like FB or Twitter')
st.write('Select your desired algo from the dropdown in the left pane')
#st.write('#1 Translation to Spanish')
#st.write('#2 Completing your sentences as you type')
#st.write('#3 Chatbot: Answering card questions')
#st.write('#4 Deduct whether your post matches FB or Twitter sub-reddit')

page = st.sidebar.selectbox(
'Select a page:',
('Chatbot:  Answering card questions', 'Translate to Spanish', 'Complete sentences',  'FB or Twitter', 'About')
)

if page == 'About':
    st.write('This site is about presenting **Data Science** magic through Streamlit!')
    st.write('Check out different selection options in the sidebar on the left')
    st.write('Thanks for visiting!')

if page == 'FB or Twitter':
    st.image('./FBTwitter/FBTwitter.jpg', width=500)

    with open('./FBTwitter/model.p', mode='rb') as pickle_in:
        pipe = pickle.load(pickle_in)
        user_text = st.text_input('Please write a sample post in the box below and we will predict below if it matches FB or Twitter subreddit:', value = "Love Facebook Groups")

    fbot = pipe.predict([user_text])[0]
    st.write(f'Your post comes from ** {fbot} **')
    st.write('Approach: FB and Twitter recent sub-reddits have been used to train an NLP model which then uses your post components to predict the sub-reddit class')

### BELOW SECTION IS FOR THE TRANSLATION TO SPANISH ####
if page == 'Translate to Spanish':

    # Encoder training setup
    num_encoder_tokens = 3917
    num_decoder_tokens = 5899

    max_encoder_seq_length = 7
    max_decoder_seq_length = 15

    latent_dim = 256

    ## Reading the various dictionaries
    with open('./TranslateEngSpan/GCP_model/data/rtfd.p', 'rb') as fp:
        reverse_target_features_dict = pickle.load(fp)

    with open('./TranslateEngSpan/GCP_model/data/tfd.p', 'rb') as fp:
        target_features_dict = pickle.load(fp)

    with open('./TranslateEngSpan/GCP_model/data/rifd.p', 'rb') as fp:
        reverse_input_features_dict = pickle.load(fp)

    with open('./TranslateEngSpan/GCP_model/data/ifd.p', 'rb') as fp:
        input_features_dict = pickle.load(fp)


    #TRANSLATING UNSEEN TEXT
    from tensorflow.keras.models import Model, load_model

    training_model = load_model('./TranslateEngSpan/GCP_model/models/training_model_gcp.h5')

    encoder_inputs = training_model.input[0] #input1
    encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output #lstm1
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)

    latent_dim = 256

    ## NEW
    decoder_inputs = training_model.input[1] #input2

    ##
    decoder_state_input_hidden = Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_cell = Input(shape=(latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    ## NEW
    decoder_lstm = training_model.layers[3]
    ##


    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]

    ## NEW
    decoder_dense = training_model.layers[4]
    ##

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    #################################################*********************

    def string_to_matrix(user_input):
        '''This function takes in a string and outputs the corresponding matrix'''
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    #######################################################*************

    def decode_sequence(test_input):
        '''This function takes in a sentence and returns the decoded sentence'''

        # Encode the input as state vectors.
        states_value = encoder_model.predict(string_to_matrix(test_input))

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0, target_features_dict['<START>']] = 1.

        # Sampling loop for a batch of sequences
        decoded_sentence = ''

        stop_condition = False
        while not stop_condition:
            # Run the decoder model to get possible output tokens (with probabilities) & states
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

            # Choose token with highest probability and append it to decoded sentence
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]

            # Exit condition: either hit max length or find stop token.
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
            else:
                decoded_sentence += " " + sampled_token


            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [hidden_state, cell_state]

        return decoded_sentence

    #######################################***********************************************

    # user inputs
    st.title('English to Spanish translator')
    question_text_eng=''
    question_text_eng = st.text_input("Hola!  Please enter the text you would like to be translated to Spanish and hit 'Enter'")

    # processing inputs for nlp model
    translated_text = decode_sequence(question_text_eng)
    if question_text_eng != '':
        st.write(translated_text)

    st.write('***  The text is translated by leveraging tokenization, lemmatization for NLP and LSTM-based Neural Net models ***')

    ##############################################################
    ## BELOW SECTION IS FOR AGENTS TO GET HELP FOR QUESTIONS

if page == 'Chatbot:  Answering card questions':
    #header
    st.title('Basic Chatbot for Card Customers')

    # user inputs
    question_text =''
    st.write('We can help with questions related to the card application process, payments, rewards, & credit bureau')
    question_text = st.text_input("How can we help you?  Please type your question (ex: 'How do i apply for a card?') below and hit 'Enter'")

    #Read in the Answer Corpus
    ans_file = pd.read_csv('./COF_CS_Chat/data/IntentAnswers.csv')
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
    cs_model = pickle.load(open('./COF_CS_Chat/models/cs_model.p', 'rb'))

    # processing inputs for nlp model
    input_text = preprocess_nlp(question_text)

    ip_series = pd.Series(input_text)

    answer_nlp = cs_model.predict(input_text)

    if question_text != '':
        ans.loc[answer_nlp[0]][0]
    st.write('***  The answer is generated using NLP technics and classification algos **')
    st.write('Opportunities to improve chatbot ability by feeding real chats to train the model better"')

    ##############################################################
### BELOW SECTION IS FOR COMPLETING SENTENCES ####
if page == 'Complete sentences':

    # Encoder training setup
    num_encoder_tokens = 5286
    num_decoder_tokens = 5264

    max_encoder_seq_length = 9
    max_decoder_seq_length = 8

    latent_dim = 256

    ## Reading the various dictionaries
    with open('./SentenceCompletion/GCP_model/data/rtfd.p', 'rb') as fp:
        reverse_target_features_dict = pickle.load(fp)

    with open('./SentenceCompletion/GCP_model/data/tfd.p', 'rb') as fp:
        target_features_dict = pickle.load(fp)

    with open('./SentenceCompletion/GCP_model/data/rifd.p', 'rb') as fp:
        reverse_input_features_dict = pickle.load(fp)

    with open('./SentenceCompletion/GCP_model/data/ifd.p', 'rb') as fp:
        input_features_dict = pickle.load(fp)


    #TRANSLATING UNSEEN TEXT
    from tensorflow.keras.models import Model, load_model

    training_model_c = load_model('./SentenceCompletion/GCP_model/models/training_model_gcp.h5')

    encoder_inputs = training_model_c.input[0] #input1
    encoder_outputs, state_h_enc, state_c_enc = training_model_c.layers[2].output #lstm1
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)

    latent_dim = 256

    ## NEW
    decoder_inputs = training_model_c.input[1] #input2

    ##
    decoder_state_input_hidden = Input(shape=(latent_dim,), name="input_3")
    decoder_state_input_cell = Input(shape=(latent_dim,), name="input_4")
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]

    ## NEW
    decoder_lstm = training_model_c.layers[3]
    ##

    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]

    ## NEW
    decoder_dense = training_model_c.layers[4]
    ##

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    #################################################*********************

    def string_to_matrix(user_input):
        '''This function takes in a string and outputs the corresponding matrix'''
        tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
        user_input_matrix = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
        for timestep, token in enumerate(tokens):
            if token in input_features_dict:
                user_input_matrix[0, timestep, input_features_dict[token]] = 1.
        return user_input_matrix

    #######################################################*************

    def decode_sequence(test_input):
        '''This function takes in a sentence and returns the decoded sentence'''

        # Encode the input as state vectors.
        states_value = encoder_model.predict(string_to_matrix(test_input))

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first token of target sequence with the start token.
        target_seq[0, 0, target_features_dict['<START>']] = 1.

        # Sampling loop for a batch of sequences
        decoded_sentence = ''

        stop_condition = False
        while not stop_condition:
            # Run the decoder model to get possible output tokens (with probabilities) & states
            output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)

            # Choose token with highest probability and append it to decoded sentence
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_token = reverse_target_features_dict[sampled_token_index]

            # Exit condition: either hit max length or find stop token.
            if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True
            else:
                decoded_sentence += " " + sampled_token


            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [hidden_state, cell_state]

        return decoded_sentence

    #######################################***********************************************

    # user inputs
    st.title('Complete your sentences')
    question_text_comp=''
    question_text_comp = st.text_input("Please enter the first few words of sentence you would like to complete and hit 'Enter'")

    # processing inputs for nlp model
    completed_text = decode_sequence(question_text_comp)
    if question_text_comp != '':
        st.write(completed_text)

    st.write('***  The text is completed using LSTM-based Neural Net models ***')

    ##############################################################
