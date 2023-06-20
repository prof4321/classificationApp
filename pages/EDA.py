import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
# from Projects import read_df

# Read the DataFrame from a CSV file

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
df = read_df("resources/data/dataset.csv")


@st.cache_data
def perform_eda(df, n):
    """
    Performs basic exploratory data analysis on a DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to analyze.
        n (int): Index of the row to display.

    Returns:
        dict: EDA results.
    """
    total_categories = len(df)
    if n >= total_categories:
        return {}
    row = df.iloc[n]
    category = row['category']
    text = row['description']
    
    return {
        "Total number of Categories": total_categories,
        "Row Number": n,
        "Category": category,
        "Text": text
    }

@st.cache_data
def generate_bar_chart(df):
    plt.style.use('ggplot')
    num_classes = len(df["category"].value_counts())
    colors = plt.cm.Dark2(np.linspace(0, 1, num_classes))
    iter_color = iter(colors)
    fig, ax = plt.subplots(figsize=(20,12))
    df["category"].value_counts().plot.barh(title="Reviews for each Category (n, %)", 
                                                    ylabel="Categories",
                                                    color=colors,
                                                    ax=ax)
    for i, v in enumerate(df["category"].value_counts()):
        c = next(iter_color)
        plt.text(v, i,
                f" {v}, {round(v*100/df.shape[0],2)}%", 
                color=c, 
                va='center', 
                fontweight='bold')

    st.pyplot(fig)

@st.cache_data
def top_correlated_terms(df, N):
    """
    Displays the top correlated unigrams and bigrams for each category in a given DataFrame.

    Parameters:
        df (pandas.DataFrame): A DataFrame containing the columns 'category' and 'description'.
        N (int): The number of top correlated terms to show.

    Returns:
        None
    """

    filtered_df = df[['category', 'description']]
    filtered_df['category_id'] = filtered_df['category'].factorize()[0]
    category_id_df = filtered_df[['category', 'category_id']].drop_duplicates()
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'category']].values)

    # Define the vectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                            ngram_range=(1, 2), 
                            stop_words='english')

    # Fit the vectorizer and transform the data
    features = tfidf.fit_transform(filtered_df.description)

    # Get the labels
    labels = filtered_df.category_id

    # Loop through the categories and find the most correlated terms
    for Category, category_id in sorted(category_to_id.items()):
        # Calculate the chi-squared test statistic for each term
        chi2score = chi2(features, labels == category_id)[0]
        # Sort the indices in decreasing order of chi-squared test statistic
        indices = np.argsort(chi2score)[::-1]
        # Get the feature names in decreasing order of chi-squared test statistic
        feature_names = np.array(tfidf.get_feature_names_out())[indices]
        # Get the top N correlated terms
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1][:N]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2][:N]
        # Print the most correlated unigrams and bigrams for each category
        st.write("\n==> %s:" %(Category))
        st.write("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams)))
        st.write("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams)))


# Set up the Streamlit app
st.sidebar.header('EDA')
diag_type = st.sidebar.selectbox('Select a display type', ['Dataset', 'Basic_EDA', 'Bar Chart', 'Text Preprocessing'])

# Display EDA plots based on the user's selection
if diag_type == 'Dataset':
    st.subheader('Dataset')
    st.write(df.head())

elif diag_type == 'Basic_EDA':
    st.subheader('Basic EDA')
    # Create a text box for the user to input the row number
    n = st.number_input("Enter the row number:", min_value=0, max_value=len(df)-1)

    # Perform EDA on the selected row
    if n >= 0 and n < len(df):
        eda_df = pd.DataFrame.from_dict(perform_eda(df, n), orient='index', columns=['Value'])
        st.table(eda_df)
    else:
        st.write("Invalid row number")

elif diag_type == 'Bar Chart':
    st.subheader('Bar Chart')
    generate_bar_chart(df)

 # Show the top correlated terms
elif diag_type == 'Text Preprocessing':
    st.subheader('Most Correlated Terms')
    # Define the minimum and maximum values for N
    N_MIN = 1
    N_MAX = 10

    # Ask the user for the value of N
    N = st.slider('Select the number of top correlated terms to show:', N_MIN, N_MAX, 3)
    top_correlated_terms(df, N)
