# import libraries
# Standard imports
import pandas as pd
import numpy as np
import re
import pickle
import string
import json
import streamlit as st

# Visualization library
import seaborn as sns
import matplotlib.pyplot as plt

# NLP library
from nltk.tokenize import word_tokenize
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.layers import Input, LSTM, Dense, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.backend import manual_variable_initialization

manual_variable_initialization(True)

#header
st.title('Quick help for Banco Uno agents')

page = st.sidebar.selectbox(
'Select a page:',
('Translate to Spanish', 'Complete sentences', 'Help for customer questions', 'About')
)

if page == 'About':
    st.write('This site is about presenting **Data Science** magic through Streamlit!')
    st.write('Check out different selection options in the sidebar on the left')
    st.write('Thanks for visiting!')

### BELOW SECTION IS FOR THE TRANSLATION TO SPANISH ####
if page == 'Translate to Spanish':

    # header image
    #col1, col2 = st.beta_columns(2)
    #image1 = Image.open('./streamlit_images/field.jpeg')
    #col1.image(image1, use_column_width=True)
    #image2 = Image.open('./streamlit_images/ginger.jpeg')
    #col2.image(image2, use_column_width=True)

    # Encoder training setup
    num_encoder_tokens = 5134
    num_decoder_tokens = 8263

    max_encoder_seq_length = 8
    max_decoder_seq_length = 17

    latent_dim = 256

    ## Reading the various dictionaries
    with open('./data/rtfd.p', 'rb') as fp:
        reverse_target_features_dict = pickle.load(fp)

    with open('./data/tfd.p', 'rb') as fp:
        target_features_dict = pickle.load(fp)

    with open('./data/rifd.p', 'rb') as fp:
        reverse_input_features_dict = pickle.load(fp)

    with open('./data/ifd.p', 'rb') as fp:
        input_features_dict = pickle.load(fp)


    #TRANSLATING UNSEEN TEXT
    from tensorflow.keras.models import Model, load_model

    training_model = load_model('./models/training_model_gcp.h5')

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
    question_text = st.text_input("Please enter the text you would like to be translated to Spanish")

    # processing inputs for nlp model
    translated_text = decode_sequence(question_text)
    st.write(translated_text)
    
    st.write('***  The text was translated using LSTM-based Neural Net models ***')
    
