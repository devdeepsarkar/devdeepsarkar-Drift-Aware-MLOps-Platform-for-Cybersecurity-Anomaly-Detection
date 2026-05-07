import os
import streamlit as st
import pandas as pd
import requests
import io

# API endpoint configuration — overridden via env var when running in Docker
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# 41 feature columns — the only ones the model uses for prediction
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Alias used by the Single Prediction tab
COLUMNS = FEATURE_COLUMNS

st.set_page_config(page_title=" Cybersecurity Log Anomaly Detection", layout="wide")

# Custom CSS for more cyberpunk/cybersec colors
st.markdown("""
<style>
    /* 1. Typing Animation for Title */
    .typing-title {
        overflow: hidden;
        white-space: nowrap;
        border-right: 0.15em solid #00FF41;
        width: 48ch;
        animation: 
            typing 2s steps(48, end),
            blink-caret 0.8s step-end infinite;
        color: #00FF41 !important;
        text-shadow: 0px 0px 8px rgba(0, 255, 65, 0.4);
        font-family: monospace;
    }
    
    @keyframes typing {
        from { width: 0 }
        to { width: 48ch }
    }
    
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #00FF41; }
    }
    
    /* Neon cyan for subheaders ONLY */
    h2, h3 {
        color: #00e5ff !important;
    }
    
    /* Bright orange/yellow for metrics */
    div[data-testid="stMetricValue"] {
        color: #ffaa00 !important;
        text-shadow: 0px 0px 5px rgba(255,170,0,0.4);
    }
    
    /* 2. Fade In and Float Animation for Bento Boxes */
    @keyframes fade-in-up {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        border: 1px solid #1f3a5f !important;
        box-shadow: inset 0 0 10px rgba(0, 229, 255, 0.03);
        animation: fade-in-up 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    /* 3. Hover Effect on Bento Boxes */
    div[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        transform: translateY(-3px);
        box-shadow: inset 0 0 15px rgba(0, 229, 255, 0.1), 0 5px 15px rgba(0, 229, 255, 0.15) !important;
    }
    
    /* File uploader styling */
    section[data-testid="stFileUploadDropzone"] {
        background-color: #111a26 !important;
        border: 1px dashed #00e5ff !important;
        transition: all 0.3s ease;
    }
    section[data-testid="stFileUploadDropzone"]:hover {
        background-color: #1a2a40 !important;
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.3);
    }
    
    /* 4. Button Hover Animations */
    button[data-testid="baseButton-secondary"] {
        transition: all 0.3s ease !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.5) !important;
        border-color: #00e5ff !important;
        color: #00e5ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='typing-title'>[*] Cybersecurity Log Anomaly Detection System</h1>", unsafe_allow_html=True)
st.markdown("""
Welcome to the Drift-Aware MLOps Dashboard for Cybersecurity. 
Upload your network logs (NSL-KDD format) to detect anomalies in real-time.
""")

tab1, tab2 = st.tabs(["Batch Prediction (File Upload)", "Single Prediction"])

with tab1:
    with st.container(border=True):
        st.header("Upload Logs for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV or TXT file", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        with st.container(border=True):
            st.success("File uploaded successfully!")
            
            # Read file to display a preview
            df_preview = pd.read_csv(uploaded_file, header=None)

            # Strip label & difficulty_level if present (cols 42 and 43)
            # Keep only the first 41 feature columns regardless of input size
            if df_preview.shape[1] > len(COLUMNS):
                df_preview = df_preview.iloc[:, :len(COLUMNS)]

            if df_preview.shape[1] == len(COLUMNS):
                df_preview.columns = COLUMNS
            else:
                df_preview.columns = df_preview.columns.astype(str)

            st.subheader("Data Preview (41 features)")
            st.dataframe(df_preview.head(), use_container_width=True)
        
        with st.container(border=True):
            predict_button = st.button("Predict Anomalies", use_container_width=True)
            
        if predict_button:
            with st.spinner("Analyzing logs..."):
                # Reset file pointer
                uploaded_file.seek(0)
                
                try:
                    # Send file to API
                    files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
                    response = requests.post(f"{API_URL}/predict_batch", files=files)
                    
                    if response.status_code == 200:
                        result_data = response.json()
                        predictions = result_data.get("predictions", [])
                        labels = result_data.get("labels", ["anomaly" if p else "normal" for p in predictions])
                        
                        df_preview["Prediction"] = predictions
                        df_preview["Status"] = labels
                        
                        with st.container(border=True):
                            st.success("Analysis Complete!")
                            known   = labels.count("confirmed threat")
                            unknown = labels.count("novel threat")
                            normal  = labels.count("normal")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Logs", len(predictions))
                            with col2:
                                st.metric("[+] Normal", normal)
                            with col3:
                                st.metric("[!] Confirmed Threats", known)
                            with col4:
                                st.metric("[?] Novel Threats", unknown)
                                
                        with st.container(border=True):
                            st.subheader("Results Preview (Top 1000 rows)")
                            def highlight_status(s):
                                styles = []
                                for v in s:
                                    if v == "confirmed threat":
                                        styles.append('background-color: #8A0E0E; color: white; font-weight: bold;')
                                    elif v == "novel threat":
                                        styles.append('background-color: #8A4E00; color: white; font-weight: bold;')
                                    else:
                                        styles.append('')
                                return styles
                            
                            pd.set_option("styler.render.max_elements", 10000000)
                            st.dataframe(df_preview.head(1000).style.apply(highlight_status, subset=['Status']), use_container_width=True)
                            st.caption("Displaying the top 1000 rows. Full annotated dataset available below.")
                            
                            csv = df_preview.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Annotated Results",
                                data=csv,
                                file_name='annotated_logs.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
                    else:
                        st.error(f"API Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: Is the API running? ({e})")

with tab2:
    with st.container(border=True):
        st.header("Single Log Prediction")
        st.markdown("Enter comma-separated features for a single network log:")
        
        # Default example from KDDTrain (Normal)
        default_features = '0,tcp,http,SF,232,8153,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,5,5,0.20,0.20,0.00,0.00,1.00,0.00,0.00,30,255,1.00,0.00,0.03,0.04,0.03,0.01,0.00,0.01'
        
        user_input = st.text_area("Features (41 values)", value=default_features)
        
        if st.button("Predict Single", use_container_width=True):
            try:
                # Parse input
                features = [x.strip() for x in user_input.split(",")]
                
                # Convert numeric strings to actual numbers
                parsed_features = []
                for f in features:
                    try:
                        if '.' in f:
                            parsed_features.append(float(f))
                        else:
                            parsed_features.append(int(f))
                    except ValueError:
                        parsed_features.append(f) # Keep as string if it can't be parsed
                        
                if len(parsed_features) != 41:
                    st.warning(f"Expected 41 features, but got {len(parsed_features)}.")
                else:
                    with st.spinner("Predicting..."):
                        response = requests.post(f"{API_URL}/predict", json={"features": parsed_features})
                        
                        if response.status_code == 200:
                            result = response.json()
                            label = result.get("label", "")
                            with st.container(border=True):
                                if label == "confirmed threat":
                                    st.error("[!] Confirmed Threat — Known Attack Pattern (Random Forest)")
                                    st.caption("This traffic matches a known attack signature in the training dataset.")
                                elif label == "novel threat":
                                    st.warning("[?] Novel Threat — Potential Zero-Day (Autoencoder)")
                                    st.caption("Unusual traffic that doesn't match any known attack. Possible Zero-Day.")
                                else:
                                    st.success("[+] Normal Traffic")
                                    st.caption("Both Random Forest and Autoencoder classify this as normal traffic.")
                        else:
                            st.error(f"API Error: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
