import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import preprocessing as kprocessing
from tensorflow.keras import models, layers, optimizers
# text preprocessing modules
from nltk.tokenize import word_tokenize
 
import nltk
from nltk.corpus import stopwords

import re  # regular expression

# Define a function to load the data
@st.cache_data
def read_df(file_path):
    """
    Reads a CSV file and returns a filtered DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Filtered DataFrame.
    """
    df = pd.read_csv(file_path, encoding='utf-8')
    df = df[['Category', 'meta_description']]
    df = df.dropna(subset=['meta_description'])
    
    # Renaming second column for a simpler name
    df.columns = ['category', 'description']

    return df


@st.cache_data
def filter_top_categories(df):
    """
    Filters a DataFrame to include only the top 15 categories by count.

    Args:
        df (pandas.DataFrame): DataFrame to filter.

    Returns:
        pandas.DataFrame: Filtered DataFrame.
    """
    category_counts = df['category'].value_counts()
    top_categories = category_counts[:15].index.tolist()
    if category_counts.nunique() <= 15:
        filtered_df = df
    else:
        filtered_df = df[df['category'].isin(top_categories)]
    return filtered_df

# Read the DataFrame from a CSV file
file_path = "resources/data/dataset.csv"
df = read_df(file_path)
df = df.reset_index(drop=True)

# Filter the DataFrame to include only the top categories
filtered_df = filter_top_categories(df)


#define load models
@st.cache_resource
def load_models():
    # Load the BERT model
    path = "resources/models/best_model.pt"
    if torch.cuda.is_available():
        state_dict = torch.load(path)
    else:
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=13)
    bert_model.load_state_dict(state_dict)

    # Load the LR model
    lr_model = pickle.load(open('resources/models/lr.pkl', 'rb'))

    # Load the LSTM model
    lstm_model = load_model('resources/models/MCTC.h5')

    return bert_tokenizer, bert_model, lr_model, lstm_model

bert_tokenizer, bert_model, lr_model, lstm_model = load_models()


# Define the Streamlit app
def main():

    # Sidebar for model selection
    with st.sidebar:
        selected = option_menu ('Multiple ML Prediction Models',
                                ['BERT', 'LR', 'LSTM'],
                                default_index=0)

    # display the front end aspect
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Company's classification using {} Model</h1> 
    </div> 
    """.format(selected)
    st.markdown(html_temp, unsafe_allow_html = True)


    # BERT Model page
    def bert_prediction(text):
        label_map = {'Healthcare': 0, 'Commercial Services & Supplies': 1, 'Transportation & Logistics': 2, 'Materials': 3, 'Corporate Services': 4, 'Information Technology': 5, 'Financials': 6, 'Professional Services': 7, 'Consumer Discretionary': 8, 'Consumer Staples': 9, 'Media, Marketing & Sales': 10, 'Industrials': 11, 'Energy & Utilities': 12}
        class_names = list(label_map.keys())
        inputs = bert_tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=35)
        outputs = bert_model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1)
        predicted_class_name = class_names[predicted_class.item()]
        return predicted_class_name

    if (selected == 'BERT'):
        text = st.text_input('Company Description', 'Type here')
        if st.button("BERT Predict"): 
            result = bert_prediction(text)   
            st.success("**This Company Description is for the class {}**".format(result))


    # LR Model page
    elif (selected == 'LR'):
        # page title
        # st.title("Company's classification using LR Model")

        # Display a dropdown menu to select a company description
        options = filtered_df['description'].tolist()
        random_10 = pd.DataFrame(options).sample(n=10)
        selected_option = st.selectbox('***Select a Company Description:***', options)

        # Predict the category based on the selected description
        predicted_category = lr_model.predict([selected_option])[0]

        # Display the predicted category
        st.write('**Predicted Category:**', predicted_category)

        # Find the row in the DataFrame that matches the predicted category
        filtered_row = filtered_df[filtered_df['description'] == selected_option]

        # Display the category and description of the selected company
        st.write('**Actual Category:**', filtered_row['category'].values[0])

    # LSTM Model page
    elif (selected == 'LSTM'):
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        def clean_text(text):
            """
                text: a string
                
                return: modified initial string
            """
            text = text.lower() # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
            text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
            text = text.replace('x', '')
        #    text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
            return text
        
        df['description'] = df['description'].apply(clean_text)
        # Load the tokenizer
        tokenizer = Tokenizer(num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(df['description'].values)
        MAX_SEQUENCE_LENGTH = 250 # set the maximum sequence length
        
        new_description = st.text_input('Enter the Company description:')
        
        if st.button('Classify'):
            seq = tokenizer.texts_to_sequences([new_description])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
            pred = lstm_model.predict(padded)
            labels = ['Commercial Services & Supplies', 'Consumer Discretionary',
                    'Consumer Staples', 'Corporate Services', 'Energy & Utilities',
                    'Financials', 'Healthcare', 'Industrials', 'Information Technology',
                    'Materials', 'Media, Marketing & Sales', 'Professional Services',
                    'Transportation & Logistics']
            st.write('Prediction:', labels[np.argmax(pred)])
        
    
if __name__ == '__main__':
    main()
