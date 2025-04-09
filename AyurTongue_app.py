import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import io
from fpdf import FPDF
from streamlit_cropper import st_cropper
from datetime import datetime
import base64
from rembg import remove
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB3


# Set up Streamlit page configuration
st.set_page_config(
    page_title="Tongue Diagnostic Device - Indian Traditional Medicine",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Define class labels
class_labels = ['Kapha', 'Pitta', 'Vata']

# Function to preprocess the image
def preprocess_image(img):
    img_resized = img.resize((224, 224)).convert('RGB')
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Predict the class of the image
def predict_image(img_path):
    img_array = preprocess_image(img_path)
    base_model = EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')

    features = base_model.predict(img_array)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    #print(f"Predicted Class: {class_labels[predicted_class]}")
    return predicted_class



@st.cache_resource
def load_model():
    return tf.keras.models.load_model("tongue_classification_model.h5")

model = load_model()
#[theme]
base="dark"
primaryColor="purple"
font="serif"
logo_path = "ayurtongue logo.png"
# Mock AI prediction function


# Optimize background removal
def optimized_remove_background(image):
    image_resized = image.resize((512, 512))  # Resize for faster processing
    image_np = np.array(image_resized)
    output_np = remove(image_np)
    return Image.fromarray(output_np)
# Function to generate PDF report
# Function to generate PDF report
# Function to generate PDF report
def generate_pdf(name, age, gender, image, result):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.image(logo_path, x=10, y=8, w=30)
    pdf.ln(7)
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "                       AyurTongue Diagnosis", ln=True)
    pdf.ln(7)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(95, 10, "DIAGNOSTIC REPORT", ln=True)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}  ", ln=True)
    pdf.cell(200, 10, f"Time: {datetime.now().strftime('%H:%M:%S')}", ln=True)
# Create Table with Diagnosis Details
    pdf.ln(7)
    
    if result == 0:
        Condition = "Kapha imbalance"
    elif result == 1:
        Condition = "Pitta imbalance"
    else:
        Condition = "Vata imbalance"

    pdf.set_font("Arial", size=12)
    attributes = ["Name", "Age", "Gender", "Condition"]
    values = [name, str(age), gender,  Condition]
    
    for attribute, value in zip(attributes, values):
        pdf.cell(95, 10, attribute, 1)
        pdf.cell(95, 10, value, 1, 1)
    pdf.ln(7)
    if result == 0:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(95, 10, "Dietary Recommendation:",ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(95, 10, "- Avoid excess milk, cheese, yogurt, and ice cream,", ln=True)
        pdf.cell(95, 10, "- Eat warm, cooked foods instead of cold, heavy, or oily meals.", ln=True)
        pdf.cell(95, 10, "- Limit Sweet and Salty Foods", ln=True)
        pdf.cell(95, 10, "- Use cumin, turmeric, cinnamon, and cloves to enhance metabolism.", ln=True)
        pdf.cell(95, 10, "- Limit red meat and opt for plant-based protein sources.", ln=True)
    elif result == 1:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(95, 10, "Dietary Recommendation:",ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(95, 10, "- Avoid Excess Spices and Heat", ln=True)
        pdf.cell(95, 10, "- Favor Sweet, Bitter, and Astringent Tastes", ln=True)
        pdf.cell(95, 10, "- Reduce Oily and Fried Foods", ln=True)
        pdf.cell(95, 10, "- Consume Healthy Fats (small amounts of ghee, olive oil, or coconut oil)", ln=True)
        pdf.cell(95, 10, "- Minimize Salt and Sour Tastes (pickles, and fermented foods)", ln=True)
    else:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(95, 10, "Dietary Recommendation:",ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(95, 10, "- Regular Meal Times", ln=True)
        pdf.cell(95, 10, "- Include warm soups, stews, porridges, and herbal teas", ln=True)
        pdf.cell(95, 10, "- Use a moderate amount of salt and fermented foods like yogurt or buttermilk.", ln=True)
        pdf.cell(95, 10, "- Opt for legumes, lentils, mung beans.", ln=True)
        pdf.cell(95, 10, "- Avoid carbonated beverages and excessive caffeine.", ln=True)
    

    # Special handling for Suggestions to span across both columns
    

    # Add Cropped Image
    pdf.ln(10)
    pdf.set_font("Arial", style='B', size=14)
    pdf.cell(200, 10, "Tongue Image", ln=True)
    img_path = "temp_cropped_image.jpg"
    image.save(img_path)
    pdf.image(img_path, x=30, w=50)

    return pdf


# Display logo on every page
#st.image(logo_path, width=150)
# Title and Description
def get_base64_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

data_uri = get_base64_image(logo_path)
st.markdown(f"""
    <div style='display: flex; justify-content: center; align-items: center;'>
        <img src='data:image/png;base64,{data_uri}' style='height:130px; margin-right: 20px;'>
        <h1>AyurTongue Diagnosis</h1>
    </div>
""", unsafe_allow_html=True)
#st.title("AyurTongue Diagnosis")

st.write("\nWelcome to Ayurvedic tongue diagnosis! Your tongue reveals important clues about your health and balance. With a closer look, you can learn how to improve your well-being naturally. Find simple and effective ways to bring your body and mind back into harmony. Start your journey to better health today with this holistic approach.")

# Sidebar for Navigation
#st.sidebar.image(logo_path, width=100)
#st.sidebar.title("Navigation")
st.sidebar.markdown(f"""
    <div style='display: flex; align-items: center;'>
        <img src='data:image/png;base64,{get_base64_image(logo_path)}' style='height:90px; margin-right: 10px;'>
        <h2>Navigation</h2>
    </div>
""", unsafe_allow_html=True)
nav_options = ["Home", "Upload Image", "Settings"]
choice = st.sidebar.radio("Go to:", nav_options)

if choice == "Home":
    #st.header("Welcome to the Diagnostic Interface")
    st.image("AyurTonguee.png", caption="Indian Traditional Medicine Diagnostics", use_container_width=True)
    st.write("\nThis device integrates modern AI technology with the time-tested principles of Indian traditional medicine. By analyzing your tongue, it provides insights into your health condition, identifies dosha imbalances (Vata, Pitta, Kapha).")
    #st.write("\nKey Features:")
    st.markdown("- **Dosha Analysis**: Detect imbalances in Vata, Pitta, and Kapha doshas.")
    st.markdown("- **Customized Remedies**: Receive herbal, dietary, and lifestyle suggestions tailored to your diagnosis.")
    st.markdown("- **Report History**: Access previous diagnostic reports for tracking health improvements.")
    st.markdown("- **Integration**: Combines traditional wisdom with AI-powered analysis for accurate results.")
    st.write("\nStart your journey towards better health by uploading your tongue image now!")

elif choice == "Upload Image":
    st.header("Upload Tongue Image")
    uploaded_file = st.file_uploader("Upload a tongue image (JPEG/PNG)", type=["jpeg", "jpg", "png"])
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=18, max_value=80)
    gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
    if uploaded_file is not None and name:
        # Display uploaded image
        image = Image.open(uploaded_file)
        #st.image(image, caption="Uploaded Image", use_container_width=True)
        # Allow manual cropping using Streamlit Cropper
        st.write("**Option to Manually Crop the Image**")
        cropped_image = st_cropper(image, aspect_ratio=None)
        st.image(cropped_image, caption="Cropped Image", width= 250)
        
        bg_removed_image = optimized_remove_background(cropped_image) 
        st.image(bg_removed_image, caption="Segmented Image", width= 250)

        # Diagnose button
        if st.button("Diagnose"):
            with st.spinner("Processing the image..."):
                # Path to the image
                #img_path = r"C:\Users\saray\Downloads\C1.jpeg"
                

                result = predict_image(bg_removed_image)
                st.success("Diagnosis Complete")
                #st.write("result:",result)
                st.subheader("Diagnostic Results")
                if result == 0:
                    #st.write("The tongue is of Kapha type")
                    #result=result0(cropped_image)
                    # Display Results
                    #st.write(f"**Health Condition:** {result['Health Condition']}")
                    #st.write(f"**Suggestions:** {result['Suggestions']}")
                    st.write(f"**Condition:** Kapha imbalance")
                    st.write("**Dietary Recommendation:**")
                    st.write("""
                            - Avoid excess milk, cheese, yogurt, and ice cream
                            - Eat warm, cooked foods instead of cold, heavy, or oily meals.
                            - Limit Sweet and Salty Foods
                            - Use cumin, turmeric, cinnamon, and cloves to enhance metabolism.
                            - Limit red meat and opt for plant-based protein sources.
                             """)
                elif result == 1:
                    #st.write("The tongue is of Pitta type")
                    st.write(f"**Condition:** Pitta imbalance")
                    st.write("**Dietary Recommendation:**")
                    st.write("""
                    - Avoid Excess Spices and Heat
                    - Favor Sweet, Bitter, and Astringent Tastes
                    - Reduce Oily and Fried Foods
                    - Consume Healthy Fats (small amounts of ghee, olive oil, or coconut oil)
                    - Minimize Salt and Sour Tastes (pickles, and fermented foods) 
                    """)
                else:
                    #st.write("The tongue is of Vata type")
                    #- Include sweet, sour, and salty tastes
                    st.write(f"**Condition:** Vata imbalance")
                    st.write("**Dietary Recommendation:**")
                    st.write("""
                            - Regular Meal Times
                            - Include warm soups, stews, porridges, and herbal teas
                            - Use a moderate amount of salt and fermented foods like yogurt or buttermilk.
                            - Opt for legumes, lentils, mung beans.
                            - Avoid carbonated beverages and excessive caffeine.""")
                

                
                # Generate PDF
                pdf = generate_pdf(name, age, gender, cropped_image, result)
                pdf_output = io.BytesIO()
                pdf_output_data = pdf.output(dest='S').encode('latin1')
                pdf_output.write(pdf_output_data)
                pdf_output.seek(0)
                st.download_button(label="Download Report as PDF", data=pdf_output, file_name="diagnostic_report.pdf", mime="application/pdf")



elif choice == "Settings":
    st.header("Settings")
    st.write("Customize your diagnostic device interface.")
    st.checkbox("Enable Multi-language Support (coming soon)")
    st.checkbox("Enable Voice Assistant (coming soon)")

# Footer
st.sidebar.write("\n---")
st.sidebar.write("Developed by The Girlss, 2025")
