import streamlit as st
import requests
import os
from PIL import Image

# === Seasonal Palettes ===
season_hex_palette = {
    "Dark": [
        "#853506",
        "#032c43",
        "#024332",
        "#765a15",
        "#38361a",
        "#2e2b2e",
        "#48005a",
        "#85230c",
        "#3c211e",
    ],
    "Medium": [
        "#eb5a24",
        "#2687bd",
        "#58948b",
        "#d8a109",
        "#6a795b",
        "#827f82",
        "#905099",
        "#ff575d",
        "#886244",
    ],
    "Light": [
        "#f6dac8",
        "#bee5fc",
        "#add3c3",
        "#ffeec5",
        "#d0dbc9",
        "#e4e0e1",
        "#e4e0e1",
        "#ffe5eb",
        "#eddfd1",
    ],
}

# === Tone Display Names ===
tone_display_map = {
    "Dark": "Golden Earth",
    "Medium": "Warm Neutral",
    "Light": "Soft Radiant",
}

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="Melatone: Smart Skin Tone Advisor", page_icon="🎨", layout="wide"
)
st.markdown(
    """
    <h1 style='text-align: center; color: #ff4b4b; font-size: 3rem;'>🎨Melatone: Smart Skin Tone Advisor</h1>
    <h4 style='text-align: center; color: #444;'>Discover Your Perfect Personal Color</h4><br>
""",
    unsafe_allow_html=True,
)

# === App Tabs ===
tab1, tab2 = st.tabs(["🎨 Skin Tone Analysis", "🧪 Model Evaluation"])

with tab1:
    image_path = None
    uploaded_file = None
    source_option = st.radio(
        "Choose image input method",
        ["📁 Upload a File", "📸 Live Detection with Your Webcam"],
    )

    st.markdown(
        """
        <style>
            .stRadio label { font-size: 1.2rem; font-weight: bold; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    if source_option == "📁 Upload a File":
        uploaded_file = st.file_uploader(
            "Upload your image", type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        )
        if uploaded_file:
            image_path = "temp_uploaded.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.image(image_path, caption="Uploaded Image", width=350)
            st.session_state["image_path"] = image_path

    elif source_option == "📸 Live Detection with Your Webcam":
        if st.button("Open Camera & Take Picture"):
            st.markdown(
                "<i>Camera will open in a separate window. Press <b>'c'</b> on your keyboard to capture an image.</i>",
                unsafe_allow_html=True,
            )
            if os.path.exists("capture.jpg"):
                os.remove("capture.jpg")
            result = os.system("python live_skin_tone_demo.py")
            if result == 0 and os.path.exists("capture.jpg"):
                image_path = "capture.jpg"
                st.session_state["image_path"] = image_path
                st.success("✅ Image captured successfully.")
                st.image(image_path, caption="Captured Image", width=350)
            else:
                st.error("❌ Failed to capture image.")

    image_path = st.session_state.get("image_path", None)

    if image_path and st.button("✨ Start Analysis"):
        st.markdown("---")
        st.markdown("### 🖼️ Image Preview:")
        st.image(image_path, caption="Original Image", width=350)
        st.markdown("---")

        api_url = "http://127.0.0.1:8000/image"
        request_data = {"image": image_path}

        with st.spinner("🧠 Analyzing your personal Skin Tone..."):
            try:
                response = requests.post(api_url, json=request_data)
                if response.status_code == 200:
                    result = response.json()
                    season_name = result["season"]
                    tone_description = result["tone"]
                    display_name = tone_display_map.get(season_name, season_name)

                    st.balloons()
                    st.markdown(
                        f"<h2 style='text-align: center; color: #7D3C98;'>🌸 Skin Tone Detected: {display_name}</h2>",
                        unsafe_allow_html=True,
                    )
                    st.divider()

                    style_suggestions = {
                        "Dark": [
                            ("✔️", "Jewel tones: Emerald, Burgundy, Cobalt, Deep Teal"),
                            ("✔️", "High-contrast outfits (e.g., black & white)"),
                            ("✔️", "Bold graphic prints, color-blocking styles"),
                            (
                                "❌",
                                "Avoid muted pastels or dusty tones that wash you out",
                            ),
                        ],
                        "Medium": [
                            (
                                "✔️",
                                "Muted earthy tones: Olive, Camel, Dusty Rose, Denim Blue",
                            ),
                            ("✔️", "Warm neutrals like Terracotta or Soft Teal"),
                            ("✔️", "Layered outfits with balanced color harmony"),
                            (
                                "❌",
                                "Avoid overly dark or neon shades that overpower your features",
                            ),
                        ],
                        "Light": [
                            ("✔️", "Pastels: Peach, Baby Blue, Lavender, Mint, Ivory"),
                            ("✔️", "Low-contrast pairings (e.g., blush + beige)"),
                            ("✔️", "Light fabrics like chiffon, organza, silk"),
                            (
                                "❌",
                                "Avoid deep or harsh colors that overpower your softness",
                            ),
                        ],
                    }
                    st.subheader("🖌️ Style Tips for You")
                    for symbol, tip in style_suggestions.get(
                        season_name, [("❌", "No suggestions available.")]
                    ):
                        color = "#d4edda" if symbol == "✔️" else "#f8d7da"
                        border = "#28a745" if symbol == "✔️" else "#dc3545"
                        text_color = "#155724" if symbol == "✔️" else "#721c24"
                        st.markdown(
                            f"""
                        <div style='
                            background-color: {color};
                            border-left: 6px solid {border};
                            padding: 10px 15px;
                            margin-bottom: 10px;
                            border-radius: 6px;
                            color: {text_color};
                            font-size: 16px;'>
                            {symbol} {tip}
                        </div>""",
                            unsafe_allow_html=True,
                        )

                    st.subheader("🎨 Your Recommended Color Palette")
                    palette = result.get("palette", [])
                    rows = [palette[i : i + 3] for i in range(0, len(palette), 3)]
                    for row in rows:
                        row_html = "".join(
                            [
                                f"<div style='background-color:{color}; width:40px; height:40px; display:inline-block; margin:4px; border-radius:6px; border:1px solid #ccc;'></div>"
                                for color in row
                            ]
                        )
                        st.markdown(
                            f"<div style='text-align:center;'>{row_html}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.error("❌ Server returned an error. Please check the backend.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with tab2:
    st.subheader("📷 Capture Multiple Images to Evaluate Model Performance")
    folder_name = st.text_input(
        "Enter a folder name to save your images", value="input_your_folder_name"
    )

    if st.button("Start Evaluation Session"):
        if not folder_name.strip():
            st.warning("Please enter a valid folder name.")
        else:
            folder_path = os.path.join("results", folder_name)
            st.info(f"Launching webcam to save images to: `{folder_path}`")
            exit_code = os.system(f"python model_evaluation.py {folder_name}")
            if exit_code == 0:
                st.success("✅ Evaluation session completed.")
                summary_path = os.path.join("results", folder_name, "summary.json")
                if os.path.exists(summary_path):
                    st.markdown("### 📊 Model Prediction Summary")
                    import json

                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                    for model, counts in summary.items():
                        st.markdown(f"**🔍 {model} Predictions:**")
                        for label, count in counts.items():
                            st.markdown(f"- **{label}**: {count} images")
                        st.markdown("---")
                else:
                    st.warning(
                        "⚠️ No summary file found. Something may have failed during postprocessing."
                    )
            else:
                st.error("❌ Something went wrong running model evaluation.")
