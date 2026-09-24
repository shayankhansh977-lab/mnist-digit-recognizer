import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance

# Page Layout & Config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
        margin-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔢 Handwritten Digit Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a digit image and adjust settings to predict!</div>', unsafe_allow_html=True)

model = tf.keras.models.load_model("mnist_model.h5")

def ifi(output):
    if output == 0: return "this is zero"
    elif output == 1: return "tis is one"
    elif output == 2: return "tis is two"
    elif output == 3: return "tis is three"
    elif output == 4: return "tis is four"
    elif output == 5: return "tis is five"
    elif output == 6: return "tis is six"
    elif output == 7: return "tis is seven"
    elif output == 8: return "tis is eight"
    elif output == 9: return "tis is nine"
    else: return "invalid output"

st.sidebar.header("⚙️ Image Preprocessing Settings")
contrast_val = st.sidebar.slider("Adjust Contrast", 0.5, 3.0, 1.0, 0.1)
brightness_val = st.sidebar.slider("Adjust Brightness", 0.5, 3.0, 1.0, 0.1)
threshold_val = st.sidebar.slider("Background Threshold", 50, 200, 127, 1)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose a PNG, JPG, or JPEG file", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_val)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness_val)
        
        st.image(image, caption="Processed Uploaded Image", width=250)

with col2:
    st.subheader("🎯 Prediction Panel")
    if uploaded_file is not None:
        if st.button("🚀 Predict Digit", type="primary", use_container_width=True):
            
            img = image.convert("L")
            img = img.resize((28, 28))
            img_array = np.array(img)
            
            
            if np.mean(img_array) > threshold_val:
                img_array = 255 - img_array
                
            img_array = img_array.astype('float') / 255
            img_array = img_array.reshape(1, 784)
            
            
            preds = model.predict(img_array)[0]
            output = np.argmax(preds)
            confidence = float(preds[output]) * 100
            

            st.balloons()
            
            
            st.success(f"### Output: {ifi(output)}")
            st.metric(label="Predicted Digit Number", value=int(output))
            st.write(f"**Confidence Level:** {confidence:.2f}%")
            st.progress(int(confidence))
    else:
        st.info("Please upload an image to enable prediction.")