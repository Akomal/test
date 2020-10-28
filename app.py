


import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image


classifier = pickle.load(open("C:\\Users\\KomalA\\Downloads\\cancer-app.pkl","rb"))


def about():

    st.write(
        "There are about 1.38 million new cases and 458 000 deaths from breast cancer each year (IARC Globocan, 2008).\n"
         "Breast cancer is by far the most common cancer in women worldwide.This app is designed to detect cancerous cells.The dataset used to train this model is from Kaggle.  \n"
        )



def predictor(Area,Radius,Perimeter,concave_points):
    
   
   
    prediction=classifier.predict([[Area,Radius,Perimeter,concave_points]])
    print(prediction)
    return prediction



def main():
    
    html_temp = """
    <div style="background-color:#008080;padding:20px; margin-bottom:20px">
    <h4 style="color:white;text-align:center;">Please fill the form below</h2>

    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    
    
    Area = st.text_input("Area","Enter mean value Here", key='area')
    #Area = st.number_input('Area', format="%.4f", step=0.0001)
    #Radius = st.slider("Radius", min_value=0, max_value=100000)
    Radius = st.text_input("Radius","Enter mean value",  key = 'rad')
    #Perimeter = st.slider("Perimeter", min_value=0, max_value=100000)
    Perimeter = st.text_input("Perimeter","Enter mean value", key = 'per')
    #concave_points = st.number_input('concave_points', format="%.4f", step=0.0001)
    concave_points = st.text_input("Perimeter","Enter mean value", key = 'con')

   
    
    result=""
    if st.button("Predict"):
        result= predictor(Area,Radius,Perimeter,concave_points)
        st.success('The output is {}'.format(result))
    

if __name__=='__main__':

    st.title("Breast Cancer Detection App")
    if st.button("About"):
        about()
    
    
    main()
    
   
    
  
    
        
        
    
    
        

    
        
