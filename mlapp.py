import streamlit as st
import pandas as pd
from data_analyzer import DataAnalyzer

from PIL import Image

st.title('Anthem AutoML')
image = Image.open('Logo.png')
st.image(image, use_column_width=True)


def auto_ml():
    activities = ['EDA', 'Review', 'model', 'About us']
    option = st.sidebar.selectbox('Selection option:', activities)

    if option == 'EDA':
        st.subheader("Exploratory Data Analysis")

        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))

            if st.checkbox("Display shape"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox('Display Correlation of data various columns'):
                st.write(df.corr())

    elif option == "Review":
        st.subheader("Review Uploaded Data")
        data = st.file_uploader("Upload dataset:", type=['csv', 'xlsx', 'txt', 'json'])
        st.success("Data successfully loaded")

        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(10))
            if st.checkbox("Review data"):
                da = DataAnalyzer(dataframe=df)
                st.write(da.get_analysis())

    elif option == 'model':
        st.subheader("Model Building")

    elif option == 'About us':

        st.markdown(
            'This is an interactive web page for our ML project, feel feel free to use it.'
            )

        st.balloons()


if __name__ == '__main__':
    auto_ml()