# Load libraries
import os
import time
import streamlit as st
import torch
import torch.nn as nn
import joblib
from timm import create_model
from torchvision import transforms
from PIL import Image
import pandas as pd
import scipy.stats

# Helper functions
def resource_path(relative_path):
    return os.path.join(os.path.abspath("."), relative_path)

# Page config
st.set_page_config(
    page_title="Visual Product Verification App",
    page_icon=resource_path("icon.ico"),
    layout="centered"
)

# Table rendering
def render_table(df):
    table_html = df.to_html(index=False, border=0)
    table_html = table_html.replace("<thead>", "<thead style='text-align: left;'>")
    table_html = table_html.replace("<th>", "<th style='text-align: left;'>")
    st.markdown(f"""
        <div style="overflow-x: auto;">
            {table_html}
        </div>
    """, unsafe_allow_html=True)

# Caching loaders
@st.cache_resource
def load_label_encoder(path):
    try:
        label_encoder = joblib.load(path)
        return label_encoder
    except Exception as e:
        st.error(f"Failed to load label encoder: {e}")
        return None

@st.cache_resource
def load_model(model_path, model_name, num_labels):
    try:
        device = torch.device("cpu")
        model = create_model(model_name, pretrained=False)
        model.head.fc = nn.Sequential(
            nn.Linear(model.head.fc.in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_labels)
        )
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Access resources
label_encoder_path = resource_path('label_encoder.pkl')
model_path = resource_path('model.pth')
model_name = "mobilevit_xs"

label_encoder = load_label_encoder(label_encoder_path)
model = load_model(model_path, model_name, len(label_encoder.classes_)) if label_encoder else None

# Preprocessing images
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0)

# Evaluate images
def evaluate_images(model, label_encoder, data_dir):
    model.eval()
    device = torch.device('cpu')
    results = []
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    progress = st.progress(0)

    for i, img_name in enumerate(image_files):
        img_path = os.path.join(data_dir, img_name)
        img_tensor = preprocess_image(img_path).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            softmax_outputs = torch.nn.functional.softmax(outputs, dim=1)

        predicted_class_idx = torch.argmax(softmax_outputs, dim=1).item()
        predicted_class_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        prediction_confidence = softmax_outputs[0, predicted_class_idx].item()
        entropy = scipy.stats.entropy(softmax_outputs.cpu().numpy()[0] + 1e-10)

        results.append({
            "Filename": img_name,
            "Cue": predicted_class_label,
            "Prediction Confidence": round(prediction_confidence, 2),
            "Entropy": round(entropy, 2)
        })

        progress.progress((i + 1) / len(image_files))

    return results

# Main UI
def main():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        st.title("Visual Product Verification")
        st.write(f"In the following you can upload multiple product images of a Louis Vuitton Speedy bag.\n\n To analyze respective product cues and receive a corresponding verification score, please ensure that all images belong to the same product.")

        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.write("Initializing app...")
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
        loading_placeholder.empty()
        st.session_state.initialized = True
    else:
        st.title("Visual Product Verification")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.file_uploader(
        "Drag and drop or select up to 20 image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        if len(uploaded_files) > 20:
            st.error("Your upload exceeds 20 images.")

            if st.button("🔄 Start Over"):
            # Delete all files in temp_uploads
                upload_dir = "temp_uploads"
                if os.path.exists(upload_dir):
                    for f in os.listdir(upload_dir):
                        os.remove(os.path.join(upload_dir, f))
                # Reset session state
                for key in list(st.session_state.keys()):
                    if key not in ("uploader_key",):
                        del st.session_state[key]
            # Rerun the app
                st.session_state.uploader_key += 1
                st.rerun()

            st.stop()
        else:
            st.success(f"{len(uploaded_files)} image(s) uploaded successfully.")

        st.markdown("##### Preview of Uploaded Images")
        cols = st.columns(4)
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            with cols[idx % 4]:
                st.image(image, caption=uploaded_file.name, width=150)

        # Save uploaded files to a temporary directory
        upload_dir = "temp_uploads"
        os.makedirs(upload_dir, exist_ok=True)

        # Clear existing files to prevent mixing
        for f in os.listdir(upload_dir):
            os.remove(os.path.join(upload_dir, f))

        # Save new uploads
        image_paths = []
        for file in uploaded_files:
            temp_path = os.path.join(upload_dir, file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            image_paths.append(temp_path)

        if st.button("▶️ Start Evaluation"):
            st.subheader(f"Result for Uploaded Images")
            st.markdown("##### Image Analysis")

            results = evaluate_images(model, label_encoder, upload_dir)
            df_results = pd.DataFrame(results)
            render_table(df_results)
            st.markdown(f"###### _Notes: The prediction confidence is a model’s estimated probability for the predicted label, referred to as 'cue'. Higher values indicate stronger confidence in the prediction. Entropy is a measure of uncertainty across all classes, where lower values mean more certainty and higher values more uncertainty. With 9 possible classes, the maximum entropy score amounts to 2.197._")

            st.markdown("##### Identified Cues")
            all_labels = list(label_encoder.classes_)
            count_table = df_results['Cue'].value_counts().reindex(all_labels, fill_value=0).reset_index()
            count_table.columns = ['Cue', 'Count']
            render_table(count_table)

            st.markdown("##### Identified Cue Categories")
            label_to_category = {
                'Brandstamp Label': 'Brandstamp',
                'Brandstamp Tongue': 'Brandstamp',
                'Hardware Fastenings': 'Hardware',
                'Hardware Lock': 'Hardware',
                'Hardware Zipper': 'Hardware',
                'Material': 'Material',
                'Production Batch': 'Production Batch',
                'Receipt': 'Receipt',
            }
            all_categories = ['Brandstamp', 'Hardware', 'Material', 'Production Batch', 'Receipt']

            df_for_category = df_results[df_results['Cue'] != 'None'].copy()
            df_for_category['Category'] = df_for_category['Cue'].map(label_to_category)
            category_count = df_for_category['Category'].value_counts().reindex(all_categories, fill_value=0).reset_index()
            category_count.columns = ['Cue category', 'Count']
            render_table(category_count)

            st.subheader(f"Verification Score for Uploaded Images")
            active_categories = (category_count['Count'] >= 1).sum()

            if active_categories >= 4:
                st.success(f"✅ Score = {active_categories} \n\n High verification.")
            elif active_categories == 3:
                st.warning(f"⚠️ Score = {active_categories} \n\n Be cautious and check the details.")
            else:
                st.error(f"❌ Score = {active_categories} \n\n Low verification. Gather more information or look for alternatives.")

        if st.button("🔄 Start Over"):
            # Delete all files in temp_uploads
            upload_dir = "temp_uploads"
            if os.path.exists(upload_dir):
                for f in os.listdir(upload_dir):
                    os.remove(os.path.join(upload_dir, f))
            # Reset session state
            for key in list(st.session_state.keys()):
                if key not in ("uploader_key",):
                    del st.session_state[key]
        # Rerun the app
            st.session_state.uploader_key += 1
            st.rerun()
    else:
        st.info("Please upload image files to begin.")

if __name__ == "__main__":
    main()
