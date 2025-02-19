import numpy as np
from PIL import Image
import streamlit as st
import pickle

# Load the model
@st.cache_resource
def load_model():
    with open('calories_burnt_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Predict calories burnt
def calories_prediction(input_data):
    model = load_model()
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction

# Main function
def main():
    # Set page configuration
    st.set_page_config(page_title="Calories Burnt Prediction", page_icon="🔥", layout="wide")

    # Custom CSS for styling
    st.markdown("""
        <style>
            .stButton>button {
                background-color: #FF5733;
                color: white;
                padding: 10px 24px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }
            .stButton>button:hover {
                background-color: #E64A19;
            }
            .stNumberInput>div>div>input {
                font-size: 16px;
            }
            .stSelectbox>div>div>select {
                font-size: 16px;
            }
            .stMarkdown {
                font-size: 18px;
            }
            .created-by {
                font-size: 20px;
                font-weight: bold;
                color: #FF5733;
            }
        </style>
    """, unsafe_allow_html=True)

    # Title and description
    st.title("🔥 Calories Burnt Prediction")
    st.markdown("This app predicts the calories burnt during exercise based on your input data.")

    # Sidebar
    with st.sidebar:
        st.markdown('<p class="created-by">Created by Andrew O.A.</p>', unsafe_allow_html=True)
        
        # Load and display profile picture
        try:
            profile_pic = Image.open("prof.jpeg")  # Replace with your image file path
            st.image(profile_pic, caption="Andrew O.A.", use_container_width=True, output_format="JPEG")
        except:
            st.warning("Profile image not found.")

        st.title("About")
        st.info("This app uses a machine learning model to predict calories burnt during exercise.")
        st.markdown("[GitHub](https://github.com/Andrew-oduola) | [LinkedIn](https://linkedin.com/in/andrew-oduola-django-developer)")

    result_placeholder = st.empty()

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
        age = st.number_input("Age", min_value=10, max_value=100, value=25, help="Enter your age")
        height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170, help="Enter your height in cm")
    
    with col2:
        weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70, help="Enter your weight in kg")
        duration = st.number_input("Exercise Duration (minutes)", min_value=1, max_value=300, value=30, help="Enter exercise duration in minutes")
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=120, help="Enter your heart rate in bpm")
    
    body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, help="Enter your body temperature in Celsius")
    
    # Prepare input data for the model
    gender = 0 if gender == "Male" else 1
    input_data = [gender, age, height, weight, duration, heart_rate, body_temp]

    # Prediction button
    if st.button("Predict"):
        try:
            prediction = calories_prediction(input_data)
            
            if prediction[0] > 0:
                prediction_text = f"The predicted calories burnt during exercise is {prediction[0]:.2f} kcal"
                result_placeholder.success(prediction_text)
                st.success(prediction_text)
            else:
                prediction_text = f"An error occurred: change some information"
                result_placeholder.error(prediction_text)
                st.error(prediction_text)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            result_placeholder.error("An error occurred during prediction. Please check the input data.")

if __name__ == "__main__":
    main()
