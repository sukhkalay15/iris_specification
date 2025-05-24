import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# Load the Trained Model
with open('iris_prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


# Function to get Flower Image
def get_flower_image(species):
    images = {
        'Iris-setosa':'C:\\Users\\kaurs\\Desktop\\Data science\\iris_project\\sertosa.jpg',
        'Iris-versicolor':'C:\\Users\\kaurs\Desktop\\Data science\\iris_project\\versicolor.jpg',
        'Iris-virginica': 'C:\\Users\\kaurs\Desktop\\Data science\\iris_project\\virginica.jpg'}
    return images.get(species,None)

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

    # Display Flower Image
    # Display Flower Image
    image_path = get_flower_image(predicted_species)
    if image_path:
        if os.path.exists(image_path):  # Check if the file exists
            image = Image.open(image_path)
            st.image(image, caption=f'{predicted_species} Flower', use_column_width=True)
        else:
            st.warning(f"Image file for {predicted_species} not found at {image_path}.")
    else:
        st.warning(f"No image associated with {predicted_species}.")

# Footer
st.markdown('---')
st.markdown('Developed with ❤️ using Streamlit')