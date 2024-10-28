import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from processor import Processor
from PyPDF2 import PdfReader
import tempfile
import pandas as pd

def convert_to_dataframe(job_listing):
    """Convert job listing dictionary to a Pandas DataFrame."""
    data = {
        "Attribute": ["Role", "Experience", "Skills", "Tools", "Description"],
        "Details": [
            job_listing['role'],
            job_listing['experience'],
            ', '.join(job_listing['skills']),
            ', '.join(job_listing['tools']),
            job_listing['description']
        ]
    }
    
    return pd.DataFrame(data)

def create_streamlit_app(model):
    st.title("AI Recruiter")
    url_input = st.text_input("Enter a URL:", value="https://careers.novuna.co.uk/job/senior-data-scientist-all-locations-considered-in-leeds-jid-1486")
    uploaded_file = st.file_uploader("Upload candidate cv", type=["pdf"], accept_multiple_files=False,)
    cv_text = ''
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
                pdfcontent = PdfReader(uploaded_file.name)
                for i, page in enumerate(pdfcontent.pages):
                    content = page.extract_text()
                    if content:
                        cv_text += content        
    submit_button = st.button("Submit")

    if submit_button:
        try:
            with st.spinner('AI recuiter is working...'):
                web_laoder = WebBaseLoader(url_input)
                page_data = web_laoder.load().pop().page_content
                scrapData = model.extractJob(description=page_data)
                job_df = convert_to_dataframe(scrapData[0])
                with st.expander('Job Listing Job Listing Details'):
                    st.dataframe(job_df, hide_index=True, use_container_width=True)
                profile = model.getProfile(cv_text)
                with st.expander('Profile'):
                    # Display each attribute in a readable format
                    st.subheader("Basic Information")
                    st.write(f"**Name:** {profile['Name']}")
                    st.write(f"**Email:** {profile['email']}")
                    st.write(f"**Contacts:** {profile['contacts'] if profile['contacts'] else 'N/A'}")
                    st.write(f"**Role:** {profile['role'] if profile['role'] else 'N/A'}")
                    st.write(f"**Experience:** {profile['experience']}")

                    st.subheader("Skills")
                    if profile['skills']:
                        st.write(", ".join(profile['skills']))
                    else:
                        st.write("No skills listed.")

                    st.subheader("Tools")
                    st.write(profile['tools'] if profile['tools'] else 'N/A')

                    st.subheader("Description")
                    st.write(profile['description'])    
                st.subheader("Email Decision")
                with st.spinner('AI is taking decision...'):
                 st.text_area(label='Email', value=model.matchProfile(description=page_data,  profile=cv_text, ), height=500)
                
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    processor = Processor()
    st.set_page_config(layout="wide", page_title="AI Recruiter")
    create_streamlit_app(processor)
   
