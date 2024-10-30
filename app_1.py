import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OrdinalEncoder , PolynomialFeatures ,RobustScaler
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error , r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import plotly.express as px
pd.options.display.float_format = '{:,.2f}'.format

df_new = pd.read_csv('new_df.csv',index_col=0)
pd.options.display.float_format ='{:,.2f}'.format
st.set_page_config (page_title = 'Flight Fare' , layout = "wide" , page_icon = '📊')
st.title("Flight Fare")


Prediction  = st.tabs(['Prediction 📈'])

with Prediction[0]:
    # Load data
    data = pickle.load(open('Deployment_data_1.pkl', 'rb'))
    
    # Input Data
    with st.container():
        st.subheader("📈 Prediction")
        P_col_0, P_col_1,P_col_2  = st.columns([10,10,10])
        with P_col_0:
            airline	= st.selectbox("Select airline", ["Select"] + df_new.airline.unique().tolist())
            source = st.selectbox("Select source", ["Select"] + df_new.source.unique().tolist())
            journey_day_name = st.selectbox("Select journey_day_name", ["Select"] + df_new.journey_day_name.unique().tolist())
        with P_col_1:
            destination	= st.selectbox("Select destination", ["Select"] + df_new.source.unique().tolist())
            journey_month = int(st.number_input("journey_month", 1, 12))
            journey_day	= int(st.number_input("journey_day", 1, 31))
        with P_col_2:
            dep_hour = int(st.number_input("dep_hour", 0, 24))
            total_stops	= int(st.number_input("total_stops", 0, 5))
            duration	= int(st.number_input("duration", 1, 100000))

    with st.container():
        # New Data
        col1, col2, col3 = st.columns([40,2,10])
        with col1:
            N_data = pd.DataFrame({
                "airline":[airline],
                "source":[source],
                "destination":[destination],
                "duration":[duration],
                "total_stops":[total_stops],
                "journey_month":[journey_month],
                "journey_day":[journey_day],
                "journey_day_name":[journey_day_name],
                "dep_hour":[dep_hour] } ,  index=[0])


            # # Predict
            Predict = pickle.load(open('model_2.pkl', 'rb'))
            result = Predict.predict(N_data)
            st.write(N_data)
        with col3:
            # # Output
            if st.button("Predict"):
                st.header(f"Price: {int(result)}")
                st.balloons()
