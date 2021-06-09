import streamlit as st
import pickle

st.title('Your text from FB or Twitter sub-reddit?')

page = st.sidebar.selectbox(
'Select a page:',
('FB or Twitter', 'About')
)

#st.sidebar.text_input("What's your name?")

if page == 'About':
    st.write('This site is about presenting **Data Science** magic through Streamlit!')
    st.write('Check out different selection options in the sidebar on the left')
    st.write('Thanks for visiting!')

if page == 'FB or Twitter':
    st.image('FBTwitter.jpg', width=500)

    with open('model.p', mode='rb') as pickle_in:
        pipe = pickle.load(pickle_in)
        user_text = st.text_input('Please write a sample post in the box below and we will predict below if it matches FB or Twitter subreddit:', value = "Love Facebook Groups")

    fbot = pipe.predict([user_text])[0]
    st.write(f'Your post comes from ** {fbot} **')
    st.write('Approach: FB and Twitter recent sub-reddits have been used to train an NLP model which then uses your post components to predict the sub-reddit class')

if page == 'Contact Me':
    name = st.text_input('What is your name?')
    email = st.text_input('What is your email?')
    question = st.text_input('What is your question?')

    user_query_dict = {
        'name': name,
        'email': email,
        'Question': question
    }

    st.write("")
    st.write("")
    st.write(f'Thanks for reaching out, {name}!  I look forward to emailing you about {question} at {email}')
