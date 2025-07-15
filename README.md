# JD-Q-A

!pip install langchain langchain-community langchainhub langchain-openai langchain-chroma bs4 google-generativeai langchain-google-genai sentence-transformers==2.2.2

API_KEY = 'Enter the api key'
os.environ['GOOGLE_API_KEY'] = API_KEY

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

import os
from langchain.prompts import PromptTemplate, ChatPromptTemplate # Import ChatPromptTemplate
from langchain.chains import SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI # Import the correct class
import docx  # Import docx library outside the function
from google.colab import files # Keep the import for files

# Initialize the ChatGoogleGenerativeAI model
# Assuming you have already set up the API key in a previous cell
# llm = ChatGoogleGenerativeAI(model="gemini-pro") # Use the model initialized in cell Qyss2UXYyptQ

# Define the prompt template for a chat model
prompt_template = """
Assume you are a Technical Hiring Manager.
1. Create Dashboards for Job Description after parsing.
2. Now prepare Questions on the above each dashboard to be asked to candidate and their Answers to validate.

Here is the Job Description:
{doc_text}
"""

# Function to load and extract text from PDF or DOCX
def load_text(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    text = ''

    if ext == '.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    elif ext == '.docx':
        # from docx import Document # Remove import inside the function
        doc = docx.Document(file_path) # Use the imported docx
        for para in doc.paragraphs:
            text += para.text + '\n'
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")

    return text

# File path (can be PDF or DOCX)
file_path = "/content/Job Description _ AI Promt Engineer.pdf"  # Use the direct file path

# Load the document text
doc_text = load_text(file_path)

# Create the prompt for the chat model
prompt = ChatPromptTemplate.from_template(prompt_template)

# Chain the prompt with the language model
chain = prompt | llm

# Run the chain with the document text as input
output = chain.invoke({"doc_text": doc_text})

# Extract and clean the output
formatted_output = output.content.replace('*', '')
print(formatted_output)
