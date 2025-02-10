import streamlit as st
from educhain import Educhain, LLMConfig
from educhain.engines import qna_engine
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set page configuration at the very top of the script
st.set_page_config(page_title="Educhain Multimodal Q&A", page_icon="🔍", layout="wide")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password")
    model_options = {
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemini-2.0-flash-lite-preview-02-05": "gemini-2.0-flash-lite-preview-02-05",
        "gemini-2.0-pro-exp-02-05": "gemini-2.0-pro-exp-02-05",
    }
    model_name = st.selectbox("Select Model", options=list(model_options.keys()), format_func=lambda x: model_options[x])

    st.markdown("**Powered by** [Educhain](https://github.com/satvik314/educhain)")
    st.write("❤️ Built by [Build Fast with AI](https://buildfastwithai.com/genai-course)")

# --- Initialize Educhain with Gemini Model ---
@st.cache_resource
def initialize_educhain(api_key, model_name):
    if not api_key:
        return None  # Return None if API key is missing

    gemini_model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key
    )
    llm_config = LLMConfig(custom_model=gemini_model)
    return Educhain(llm_config)


# --- Utility Function to Display Questions (Not Used in Multimodal Q&A but kept for potential future use) ---
def display_questions(questions):
    if questions and hasattr(questions, "questions"):
        for i, question in enumerate(questions.questions):
            st.subheader(f"Question {i + 1}:")
            if hasattr(question, 'options'):
                st.write(f"**Question:** {question.question}")
                st.write("Options:")
                for j, option in enumerate(question.options):
                    st.write(f"   {chr(65 + j)}. {option}")
                if hasattr(question, 'answer'):
                    st.write(f"**Correct Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            elif hasattr(question, 'keywords'):
                st.write(f"**Question:** {question.question}")
                st.write(f"**Answer:** {question.answer}")
                if question.keywords:
                    st.write(f"**Keywords:** {', '.join(question.keywords)}")
            elif hasattr(question,'answer'):
                st.write(f"**Question:** {question.question}")
                st.write(f"**Answer:** {question.answer}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            else:
                st.write(f"**Question:** {question.question}")
                if hasattr(question, 'explanation') and question.explanation:
                    st.write(f"**Explanation:** {question.explanation}")
            st.markdown("---")

# --- Streamlit App Layout ---
st.title("🔍 Educhain Multimodal Q&A")

# --- Main Content: Multimodal Q&A ---
if not api_key:
    st.warning("Please enter your Google API Key in the sidebar to continue.")
else:
    # Initialize Educhain client with Gemini model
    educhain_client = initialize_educhain(api_key, model_name)
    if educhain_client:
        qna_engine = educhain_client.qna_engine
        st.header("Solve Doubt With Image")
        uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        doubt_prompt = st.text_area("Describe what you want the AI to do:", placeholder="e.g. 'Explain the diagram in detail'")
        detail_level = st.selectbox("Select Detail Level", ["Low", "Medium", "High"])

        if uploaded_image and doubt_prompt and detail_level:
            if st.button("Solve Doubt", key='image_button'):
                with st.spinner("Analyzing Image..."):
                    image_path = f"temp_image.{uploaded_image.name.split('.')[-1]}"
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.read())

                    solved_doubt = qna_engine.solve_doubt(
                        image_source=image_path,
                        prompt=doubt_prompt,
                        detail_level=detail_level
                    )

                    os.remove(image_path)  # Clean up temp file
                    if solved_doubt:
                        st.subheader("Solution")
                        st.write(f"**Explanation:** {solved_doubt.explanation}")
                        if solved_doubt.steps:
                            st.write("**Steps:**")
                            for i, step in enumerate(solved_doubt.steps):
                                st.write(f"{i + 1}. {step}")
                        if solved_doubt.additional_notes:
                            st.write(f"**Additional Notes:** {solved_doubt.additional_notes}")
                    else:
                        st.error("Could not process image, please try again")
    else:
        st.error("Failed to initialize Educhain. Please check your API key and model selection.")
