import streamlit as st
import requests

st.set_page_config(page_title="Image Classifier", page_icon=":camera:", layout="wide")
st.header("Image Classifier")
st.subheader("Upload an image to classify")

payload = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg"])
url = '<API-ENDPOINT-URL>'
headers = {
  'Content-Type': 'image/jpeg'
}
classify = st.sidebar.button("Classify!")

if payload is not None:
    if classify:
        with st.spinner('Classifying...'):
            response = requests.request("POST", url, headers=headers, data=payload)
            response_json = response.json()
            prediction, confidence = response_json["predicted_label"], response_json["score"]
            st.image(payload, caption=f"{prediction}", width=300)
            st.write(f'### Classified as: {prediction} with {round(confidence, 2)}% confidence')
