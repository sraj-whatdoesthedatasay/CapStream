# Capstone Project
## Agent Assistant - enabling agent efficiency using ML
#### Shashank Rajadhyaksha
#### *DSIR22221-E Capstone Project, Presented 5.12.21*


## Executive Summary

Agents in various operational sites have to do a vast variety of tasks and have to multitask using the tools they have.  These agents are the face of the company to a lot of customers and the customer experience provided by these agents is very important in driving customer's perception of the product.  

In global companies, some agents also have to service accounts of accountholders who speak other languages. So it is important that agents have multiple tools at their fingertips that enable them to do their jobs better.  In addition, better tools can drive more efficiency which can help to reduce staffing demand and drive lower costs for organizations.

This project was to develop a suite of tools (called AA for Agent Assistant) to help agents in their efforts in servicing customers.  There were 3 ML based tools that have been developed here to assist the agents in their effort:

#1 English to Spanish translator:  The intent of this tool is to enable agents to communicate with Spanish-speaking customers. This was built using English and Spanish datasets of multiple sentences.  It was built using natural language processing tools in [scikit-learn](https://scikit-learn.org/stable/) and [NLTK](https://www.nltk.org/).  A sequence-to-sequence LSTM model was built using the underlying data to develop a model to enable translation.  The model's translation accuracy was around 80% for the test and train population.

#2 Sentence Completion: The intent of this tool is to enable an agent to complete their sentences.  This would be akin to how gmail completes sentences after you have typed a few words.  The same English sentences that were used for translation were used.  However, the input feature was the first few words from the English lines and the last few words from the same line was the target for the model.  Similar to the translation algorithm, a sequence-to-sequence LSTM model was built.  The model's completion accuracy is displayed to be ~80% for the test population, however, the proposed sentence completers are not as clear. 

#3 Chatbot to help agents:  The intent of this tool is to enable an agent to type in a question and it would answer the question related to the credit card.  Using the FAQ from a bank's webpage and randomizing permutations, a chat corpus was created.  This is then used to used to train the model where a question is mapped to the intent behind the question.  Intent can range from how to apply, how to earn rewards etc.  Each intent has a unique answer and the chatbot would respond with that answer.  The model accuracy here was over 90% primarily because the chat corpus was built around the FAQ words.

Execution of training and deploying of LSTM model also had to be developed separately. Given the long run-times, additional functions were developed to continue with the epoch training after a few epoch runs.  Also a set of decoding functions had to be developed to read in the text and load that into a certain form for the model's encoder and decoder to use and translate/complete.  A streamlit app was developed for all the 3 models so that AA can be demonstrated to agents and interested organizations.

Some challenges that were faced (especially for the Translator and Completer), was that not all lines could be used due to system limitations in the training process.  This limits the training of the model and the translations/completions it can come up with.  Also the model had to be run on Google Cloud Platform in order to run over a few hours.  Both Translator and Completer needed over 50 epochs which took over 5 hours to run on Google Cloud Platform.

Looking forward, the opportunity is to improve the algorithms and use more data in conjunction to improve the quality of the translation and completions.  For the chat model, the opportunity is to train the model on actual chat conversations so that it can be trained better on unseen data and develop better response.

 
## Problem Statement

Chat agents have to perform multiple tasks simultaneously that can slow them down.   They have to cover multiple languages at times, they have look up product questions so that they can get back to the customer quickly.

**We aim to help chat operations agents** by developing tools using ML models to improve their efficiency and effectiveness.

We will use ML models to **have a translater that can translate from English to Spanish**, **a sentence completer that can finish their sentences after they have typed a few words**, and **a chatbot that can answer their product related questions**.  

---

## Datasets

### Data Collection
This project used the data for translater and sentence completer from the following source: 

Link to dataset of Spanish-English sentences: http://www.manythings.org/anki/spa-eng.zip

The initial dataset had ~128k rows with each row having the English and corresponding Spanish line.  This had been curated by lines coming from different sources.  There were about 19k duplicate English lines that were then de-duped from the dataset.  There were also sentences with fewer or very high characters as well as sentences with fewer words.  Sentences with less than 18 characters and with less than 4 words and with more than 50 characters were removed.  The lower end counts were removed to not have very sparse matrices in the eventual dataset.  The above suppressions resulted in ~80k records for the Translator and ~50k records for the Completer.

For the chatbot, the data was sourced from a primary banks' website from the FAQ page with questions and answers.  The chat corpus was developed using words from the questions and expanded to multiple questions.  Each answer was tagged with the intent and the intent was mapped to an answer.  For example, question about related to application of a card was tagged as CardApplication intent with a defined answer for that intent.

Kiva's website for past loans that were requested (https://www.kiva.org/build/data-snapshots).

The initial list had more than a million loans and the loans ranged from 2006 onwards until 2019.  Given that older loans funding rate may have had different funding rates and a very different process, the dataset was limited to relatively recent loans starting from 2015 onwards as well as undersampled for funded loans to reduce the imbalance between funded and expired (not-funded) loans.

### Data Dictionary

#### English - Spanish Translator
|Spanish Sentence	|English Sentence	|
|-	|-	|
|un perro tiene cuatro patas	|	a dog has four legs	|

#### English Sentence Completer
|English Pre	|English Post	|
|-	|-	|
|a dog has 	|	four legs	|

#### Chat Dataset
|question	|intent	|
|-	|-	|
|What information does Banco Uno® require when I apply for a credit card? 	|	Cardapply
|How can I report a lost or stolen credit card?? 	|	LostStolenCard
---


---

## Model Build and Analysis for Translator and Completer
### Data Cleaning & Pre-processing Steps
- Lemmatized, tokenized and joined words to re-form sentences
- Standard processing such as converting to lower case characters, removing punctuations, removing extra spaces and digits for English & Spanish sentences
- Examined the data and removed duplicate English sentences
- Created 2 new variables around count of characters and words for English & Spanish sentences
- For Translator, kept lines with characters between 18 and 50
- For Completer, same pruning for characters and also deleted lines with less than 4 words

   
### EDA/Processing for Translator & Completer
-  - Created 2 new variables around count of characters and words for English & Spanish sentences and reviewed distributions by count of characters and words
- For Translator, kept lines with characters between 18 and 50
- For Completer, same pruning for characters and also deleted lines with less than 4 words

### Modeling & Evaluation of LSTM models
Translator and Sentence completers need NLP based models.  These models need to store the words and their sequence that comes together to form a sentence.  

This makes sequence-2-sequence LSTM models the ideal model to use since it has to retain memory of the initial sequence to create the new sequence.  There are other options like using embedding and teacher

This modelling exercise is influenced by Francois Chollet's usage of sequence-2-sequence LSTM models.  https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html.  Google's translation (https://arxiv.org/abs/1609.08144) and Facebook's translation are also based on LSTM with multiple encoder and decoder layers, more evolved algorithms and lot of memory / computation power to build these out.

Train and Test split were created and the model was trained through multiple epochs.  The local machine was not able to handle the higher memory need and was slow.  So training was moved to the Google Cloud Platform with a Cuda instance, 8-CPU machines, 64GB RAM and 100GB disk. This made the epochs run for more lines and faster.  However, the training at times got disconnected.

So the training program was modified to store the model every 5 epochs.  In addition, other function was written to load the interim model as well as load other data required for training.  This needed converting the interim dictionaries that were created into files that would then have to be read again for the model to continue learning using the loaded model.

In addition an inference model had to be written, where one would pass the text that had to be translated as input, it would then predict the translation or sentence completion by reading the encoder and decoder configurations from the saved model - and it would then write the translated Spanish text or the completed sentence.
    

## Chat Model Build
### Data Cleaning Steps
- Parsed sampled dataset down to loans only from Kenya
- Parsed the dataset further to only the text columns needed for NLP which were:
    - TAGS, LOAN_USE & DESCRIPTION_TRANSLATED
- Removed whitespace, hyphens and hashtags from tags column so full tags will be counted as a single word when tokenized rather than separating a single tag into multiple words 
- Removed html breaks from description column
- Filled null values with empty strings to minimize null values when adding columns together
- Added text columns together in new column 'joined_text'
- Dropped remaining null values and duplicates

### Preprocessing for NLP Model
- Tokenized data to remove punctuation 
- Lemmatized data so only singular forms of words remained
- Removed English stopwords

### EDA for NLP Model
- Frequency distributions of top tags 
    - The top 3 were #parent, #womanownedbusiness, #user_favorite
- Frequency distributions of top words in all text columns combined
    - Included business, farming, farm, child
- Sentiment Analysis 
    - Did not provide much insight
    
### Modeling, Iteration & Evaluation for NLP Model
- Dataset included 51,019 observations
- X-variable: 'joined_text', y-variable: 'STATUS', {'funded' : 1, 'expired': 0}
    - Baseline score: .784
- Vectorized data using Tf-IDF Vectorizer with 2 grams and 10,000 max features
    - Did not remove numbers as they gave greater predictive power
- Split data into train and test sets using train-test-split ended up with 38,264 observations for train and 12,755 for test
- Created a binary classification model using basic Logistic Regression 
    - Results:
        - Train score:  .837
        - Test score: .821
        - Cross-value score: .816
        - Accuracy score: .82
    - Word Correlations:
        - Top 5 word combinations correlated with Funded Loans: 
            - 20 000, 30 000, kes 20, singleparent, 20
        - Top 5 word combinations correlated with Expired Loans: 
            - 100 000, 100, man, repeatborrower, repairrenewreplace
    - Interpretation:
        - Our logistic regression model scored well overall; however, it didn't perform as well with the expired loans as it did with funded loans. This is likely because the classes were unbalanced. With this model we are predicting most loans to be funded and potentially giving false hope that some loans are likely to be funded when they are not. In future iterations, we would like to include a more balanced dataset, increasing the number of expired loans to be able to predict them better. 
        - The top correlated word combinations are monetary amounts in Kenyan Shillings and we can see that lower amounts under KES 30,000 are more likely to get funded and higher amounts at KES 80,000 and above are more likely to expire. Tags are also important as singleparent, repeatborrower and repairrenewreplace are all tags.
- Developed and fitted several additional classification model iterations using various hyperparameters that were optimized through trial and error. Also created model variations using stemming and modeling text columns separately to see how well they performed. Ultimately, for this project interpretation was important, so the logistic regression was the best model and was chosen for our combination model and KivaMaxApprover app.


## Agent Assistant App using Streamlit

We were able to achieve our goal of helping Chat agents by providing them tools to enable translation, sentence completion as well as a chatbot to quickly look up product terms.  

$$$$$$$
The app is built on a modified prediction algorithm that 
- returns predictions rapidly
- predicts using both Numeric & NLP models
- minimizes false positives, by requiring a "yes" from both models to predict success
- requires only five user entries (impactful features identified during the modeling process)
- requires minimal user training to operate

We believe that this app can improve efficiency for Field Partners by eliminating the need for a skilled worker to carefully review each loan application. Then, if the KivaMaxApprover app indicates that a loan is likely to be funded, an less-experienced staff member can be assigned to finalize and post the request, while more senior resources and support can be directed towards improving applications that are unlikely to be funded in their current state. 

## Conclusions & Future Directions
The chart below illustrates how our "yes from both" approach performs on testing data. **9,325 of the 12,755 loans would be correctly identified by KivaMaxApprover as likely to be funded**, and so only the remaining 26.89% (3,430) of the loan applications would require an in-depth review by an experienced staffer. 

| Num. Prediction 	| NLP Predictions 	| Actual 	| Count 	|
|-	|-	|-	|-	|
| Expired 	| Expired 	| Expired 	| 645 	|
| Expired 	| Expired 	| Funded 	| 164 	|
| Expired 	| Funded 	| Expired 	| 536 	|
| Expired 	| Funded 	| Funded 	| 280 	|
| Funded 	| Expired 	| Expired 	| 228 	|
| Funded 	| Expired 	| Funded 	| 231 	|
| Funded 	| Funded 	| Expired 	| 1346 	|
| **Funded** 	| **Funded** 	| **Funded** 	| **9325** 	|


We focus on false positives to ensure that all "questionable" applications receive careful attention from the team. But the chart also illustrates some limitations of our approach: there are several hundred loans that one of our models correctly predicts as "funded," yet our app flags them for further, potentially unnecessary review. We aren't currently able to harness the full predictive power of both models, but with more time and exploration of other model combination techniques, we should be able to improve both our predictions and our app. 

Further, we'd like to enhance the functionality of our app so that it can automatically generate specific suggestions for loan application improvement, in addition to funding predictions.   

Accompanying presentation available [here](https://docs.google.com/presentation/d/1-TZQEaNkloXdCbY4gXBSgxWtonl4iISbT7ZoZXcAuUc/edit#slide=id.g5402930dac_0_462).

---

### File Structure

```
kiva-max-approver 
|__ early_work
|   |__ Cleaning
|      |__cr_cleaning.ipynb
|      |__rzi_01_kenya_nlp_cleaning.ipynb
|      |__rzi_02_preprocessing.ipynb
|      |__rzi_nlp_cleaning.ipynb
|      |__sr1_kiva_data.ipynb
|      |__sr2_kiva_smallfile-kenya.ipynb
|      |__sr2_kiva_smallfile.ipynb
|   |__ EDA
|       |__cr_eda_modeling.ipynb
|       |__cr_eda_viz.ipynb
|       |__cr_kenya_modeling_eda.ipynb
|       |__ps_01_fulldata_nlp_clean_eda_logreg.ipynb
|       |__rzi_early_eda_bag_of_words_modeling.ipynb
|       |__rzi_sentiment_modeling.ipynb
|       |__sr3_kiva_eda-kenya.ipynb
|       |__sr3_kiva_eda.ipynb
|       |__sr_different_scenarios.xlsx
|   |__ Modeling
|       |__ps_02_fulldata_nlp_models_tags.ipynb
|       |__ps_03_kenya_nlp_models_tags_text.ipynb
|       |__ps_04_kenya_nlp_eda_logreg_corrected.ipynb
|       |__ps_05_final_nlp_model_pickle.ipynb
|       |__ps_final_nlp_model.ipynb
|       |__rzi_03_kenya_nlp_modeling.ipynb
|       |__rzi_04_model_selection_insights.ipynb
|       |__rzi_num_kiva_modeling.ipynb
|       |__sr3_kiva_kenya_FINAL_numeric_model.ipynb
|       |__sr3_kiva_kenya_model.ipynb
|__ images
|   |__ Char_count_Distr.png
|   |__ LenderTerm_Distr.png
|   |__ Loan_Distr.png
|   |__ Month_Distr.png
|   |__ Screen Shot 2021-04-25 at 6.28.25 PM.jpeg
|   |__ app-desktop.jpeg
|   |__ app-tablet.jpeg
|   |__ app_desktop_screenshot.png
|   |__ app_tablet_screenshot.png
|   |__ combo_clf_report.png
|   |__ combo_model_results.png
|   |__ feature_importance.png
|   |__ field.jpeg
|   |__ ginger.jpeg
|   |__ kiva_loan_example.png
|   |__ nlp_clf_report.png
|   |__ nlp_positive_coefs.png
|   |__ num_conf_matrix.png
|   |__ numeric_confusion_matrix.png
|   |__ numeric_default_classifiers.png
|   |__ numerical_clf_report.png
|   |__ top_occurring.jpg
|   |__ topwords_alltext.png
|__ kma_models
|   |__ nlp_model.p
|   |__ numeric_model.p
|__ nlp_model
|   |__ 01_kenya_nlp_models.ipynb
|   |__ 02_kenya_nlp_eda_logreg.ipynb
|   |__ 03_final_nlp_model_pickle.ipynb
|__ numeric_model
|   |__ .gitignore
|   |__ Step1_Global_RecentYears_DataCreation.ipynb
|   |__ Step2_Kenya_EDA_Cleaning.ipynb
|   |__ Step3_Model_Build.ipynb
|   |__ Step4_Model_for_pickle.ipynb
|__ README.md
|__ kma_app.py
|__ presentation.pdf
|__ requirements.txt
```