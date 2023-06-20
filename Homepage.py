import streamlit as st

st.set_page_config(
    page_title="CompCat - Company Classification App",
    page_icon=":bar_chart:"
)

# Add background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add custom CSS
st.markdown(
    """
    <style>
    .header {
        font-size: 48px;
        font-weight: bold;
        color: #008080;
        margin-bottom: 30px;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #555555;
        margin-bottom: 20px;
    }
    .description {
        font-size: 18px;
        line-height: 1.6;
        color: #333333;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define the app header
st.markdown('<div class="header">CompCat</div>', unsafe_allow_html=True)

# Create a two-column layout
col1, col2 = st.columns([3, 4])

# Add the image to the right column
with col2:
    st.image("./resources/imgs/Filter-pana.png", width=440)

# Add app description to the left column
with col1:
    st.markdown(" ")
    st.markdown(" ")
    st.markdown(" ")
    st.markdown('<div class="subheader">Classify companies based on their descriptions</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">This app uses different machine learning models to classify companies based on their descriptions. It also shows some of the insights in the dataset for better understanding of the data.</div>', unsafe_allow_html=True)
