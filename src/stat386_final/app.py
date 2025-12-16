import pandas as pd
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
from importlib.resources import files
import matplotlib.pyplot as plt
import seaborn as sns
from stat386_final import viz, read, preprocess, model
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo-root/
sys.path.insert(0, str(ROOT / "src"))

@st.cache_data
def load_data(filepath):
    df = read.read_data(filepath)
    return df


@st.cache_data
def clean_data(df):
    sales_combined = preprocess.process_data(df)
    cleaned_df = preprocess.prepare_data(sales_combined)
    return sales_combined, cleaned_df


@st.cache_resource
def load_model():
    # na stuff
    with open('./data/na_model.pkl', 'rb') as f:
        na_mod = pickle.load(f)
    with open('./data/na_scaler.pkl', 'rb') as f:
        na_scaler = pickle.load(f)
    # eu stuff
    with open('./data/eu_model.pkl', 'rb') as f:
        eu_mod = pickle.load(f)
    with open('./data/eu_scaler.pkl', 'rb') as f:
        eu_scaler = pickle.load(f)
    # jp stuff
    with open('./data/jp_model.pkl', 'rb') as f:
        jp_mod = pickle.load(f)
    with open('./data/jp_scaler.pkl', 'rb') as f:
        jp_scaler = pickle.load(f)
    #other stuff
    with open('./data/other_model.pkl', 'rb') as f:
        other_mod = pickle.load(f)
    with open('./data/other_scaler.pkl', 'rb') as f:
        other_scaler = pickle.load(f)
    # global stuff
    with open('./data/global_model.pkl', 'rb') as f:
        global_mod = pickle.load(f)
    with open('./data/global_scaler.pkl', 'rb') as f:
        global_scaler = pickle.load(f)
    # return all
    return na_mod, na_scaler, eu_mod, eu_scaler, jp_mod, jp_scaler, other_mod, other_scaler, global_mod, global_scaler

def main() -> None:
    st.set_page_config(page_title="Video Game Analysis", layout="wide")
    st.title("Video Game Analysis")
    st.write(
        "This is a streamlit app designed to analyze video game sales, and make predictions on other video game sales."
    )
    
    with st.sidebar:
        st.header("Sections")
        show_cleaning = st.checkbox("Show Cleaning")
        show_analysis = st.checkbox("Do Analysis")
        make_predictions = st.checkbox('Make Predictions')
        
    # initialize the dfs and the model
    filepath = './data/game_data.csv'
    df = load_data(filepath)
    sales_combined, cleaned_df = clean_data(df)
    na_mod, na_scaler, eu_mod, eu_scaler, jp_mod, jp_scaler, other_mod, other_scaler, global_mod, global_scaler = load_model()
    areas_g = ['NA', 'EU', 'JP', 'Other', 'Global']
    areas_p = ['NA', 'EU', 'JP', 'Other', 'Global']
    platforms = df["Platform"].dropna().unique().tolist()
    genres = df["Genre"].dropna().unique().tolist()
    

    st.subheader("Data Preview")
    st.write('This is a preview of the starting data set, it has 1766 rows and 13 columns.')
    st.dataframe(df, use_container_width=True)

    if show_cleaning:
        st.subheader("Cleaning Pipeline Output")
        st.write(
            "Below is a cleaned Data Frame where we combined game information so each game only appears once"
        )
        st.dataframe(sales_combined, use_container_width=True)
        st.write(
            "We also went through a prepped the data set in order to prepare it for a model evaluation, we added dummy variables for the consoles it's on and stripped out the name column. This is just a head of the data frame, to give you an idea of what it looks like."
        )
        st.dataframe(cleaned_df.head(10), use_container_width=True)

    if show_analysis:
        st.header("Analysis Pipeline Output")
        
        st.subheader('Histograms of Sales by Genre')
        genre = st.selectbox('Genre', genres)
        area_g = f"{st.selectbox('Area', areas_g)}_Sales"
        ax_g = viz.print_genre_distribution(sales=df, genre=genre, area=area_g)
        st.pyplot(ax_g.figure)

        st.subheader('Histograms of Sales by Platform')
        platform = st.selectbox('Platform', platforms)
        area_p = f"{st.selectbox('Area', areas_p, key='area_p_select')}_Sales"
        ax_p = viz.print_platform_distribution(sales=df, platform=platform, area=area_p)
        st.pyplot(ax_p.figure)

    if make_predictions:
        st.header('Make Predictions')

        st.subheader('Enter in Parameters')
        area_predict = st.selectbox('Area', areas_p, key='area_predict_select')
        genre_predict = st.selectbox('Genre', genres, key='genre_predict_select')
        platforms_predict = st.multiselect('Select Platforms', platforms, key='platforms_predict_select')
        all_time_peak = st.number_input('All Time Player Peak', 0, 490000, 1500)
        last_30_day_avg = st.number_input('Last 30 Day Average Player Size', 0, 40000, 35)
        year = st.number_input('Year', 1983, 2016, 2000)
        
        dict_list = {
            'Name': 'Yellow',
            'Platform': platforms_predict,
            'Year': year,
            'Genre': genre_predict,
            'Publisher': 'User',
            'all_time_peak': all_time_peak,
            'last_30_day_avg': last_30_day_avg,
            'NA_Sales': 0,
            'EU_Sales': 0,
            'JP_Sales': 0,
            'Other_Sales': 0,
            'Global_Sales': 0
        }
        df_input = pd.DataFrame(dict_list)
        df_input = preprocess.process_data(df_input)
        df_predict = preprocess.prepare_data(df_input)
        df_predict = df_predict.reindex(
            columns=cleaned_df.columns,
            fill_value=0
        )
        list_cols_drop = [col for col in df_predict.columns if 'Sales' in col]
        list_cols_drop.append('Rank')
        df_predict = df_predict.drop(list_cols_drop, axis = 1)
        if area_predict == 'NA':
            best_model = na_mod
            scaler = na_scaler
        elif area_predict == 'EU':
            best_model = eu_mod
            scaler = eu_scaler
        elif area_predict == 'JP':
            best_model = jp_mod
            scaler = jp_scaler
        elif area_predict == 'Other':
            best_model = other_mod
            scaler = other_scaler
        elif area_predict == 'Global':
            best_model = global_mod
            scaler = global_scaler
        if area_predict and genre_predict and platforms_predict and all_time_peak and last_30_day_avg and year:
            if st.button('Predict Sales!'):
                prediction = model.predict(best_model=best_model, area=area_predict, new_data=df_predict, scaler=scaler)
                st.write(f'Predicted Sales for {area_predict}: {round(prediction[0], 3)} Million Copies')


if __name__ == "__main__":
    main()