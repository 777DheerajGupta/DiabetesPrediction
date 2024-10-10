# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:06:26 2024

@author: AUSU
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained model
loaded_model = pickle.load(open('C:/STUDY/ML/part-5/deploying through Streamlit/trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    # Convert input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the array for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    
    # Perform prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

# Main function for the web app
def main():
    # Title of the web app
    st.title('Diabetes Prediction Web App')

    # Getting the input data from the user
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure value")
    SkinThickness = st.text_input("Skin thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes pedigree function value")
    Age = st.text_input("Age of the person")
    
    # Code for prediction
    diagnosis = ''
    
    # When the button is clicked
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to the correct type
            input_data = [int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness),
                          int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]
            
            # Make the prediction
            diagnosis = diabetes_prediction(input_data)
        
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
    
    # Display the prediction result
    st.success(diagnosis)

# Run the main function
if __name__ == '__main__':
    main()
