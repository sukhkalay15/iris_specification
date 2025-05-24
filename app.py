import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# Load the Trained Model
with open('iris_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)



# Streamlit UI
st.set_page_config(page_title='Iris Species Prediction', 
                   page_icon='🌸',
                   layout='centered')

st.title('🌿 Iris Flower Classification')
st.markdown('Predict the species of an iris flower based on its features.')

# Input fields
sepal_length = st.slider('Sepal Length(cm)', 4.0, 8.0, 5.1)
sepal_width =  st.slider('Sepal Width(cm)', 2.0, 5.0, 3.5)
petal_length = st.slider('Petal Length(cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width(cm)', 0.1, 3.0, 0.2)

# Predict button
if st.button('Predict'):
    input_data = np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    predicted_species = model.predict(input_data)[0]
    st.success(f'🌸 Predicted Species: **{predicted_species}**')


# Footer
st.markdown('---')
st.markdown('Developed with ❤️ using Streamlit')