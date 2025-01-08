import csv
from io import StringIO
import json
import os
import tempfile

# import google.generativeai as genai
from mysql.connector import connect, Error
import openai
from openai import OpenAIError
from PyPDF2 import PdfReader
import streamlit as st
from striprtf.striprtf import rtf_to_text
import zipfile


def save_to_mysql_message(
        conn, 
        session_id1: int, 
        role1: str, 
        model1: str, 
        content1: str,
        _image_file_path: str = "",
        ) -> None:
    """
    Inserts a new message into the "message" table in a MySQL database table.

    Parameters:
    - conn (MySQLConnection): A connection object to the MySQL database.
    - session_id1 (int): The session ID associated with the message.
    - role1 (str): The role of the user sending the message.
    - model1 (str): The mdoel used.
    - content1 (str): The content of the message.
    - _image_file_path (str, optional): The path to the image file associated with the message. Defaults to "".

    Raises:
    - Error: If the message could not be saved.
    """
    try:
        with conn.cursor() as cursor:
            if _image_file_path != "":
                sql = "INSERT INTO message (session_id, role, model, content, image) VALUES (%s, %s, %s, %s, %s)"
                val = (session_id1, role1, model1, content1, _image_file_path)
                cursor.execute(sql, val)
                conn.commit()
            else:
                sql = "INSERT INTO message (session_id, role, model, content) VALUES (%s, %s, %s, %s)"
                val = (session_id1, role1, model1, content1)
                cursor.execute(sql, val)
                conn.commit()

    except Error as error:
        st.error(f"Failed to save new message: {error}")
        raise


def extract_text_from_pdf(pdf) -> str: 
    """
    Extract and concatenate text from all pages of a PDF file.

    This function reads a PDF file from the given path, extracts text from each page, 
    and concatenates it into a single string.

    Parameters:
    pdf_path (str): The file path to the PDF from which to extract text.

    Returns:
    str: A string containing all the extracted text from the PDF.
    """
    pdf_reader = PdfReader(pdf) 
    return ''.join(page.extract_text() or '' for page in pdf_reader.pages)


def extract_jason_from_csv(csv_file) -> str:
    """
    Converts an uploaded CSV file to a JSON string.

    Args:
        csv_file: An UploadedFile object representing the uploaded CSV file.

    Returns:
        str: The JSON string representation of the CSV data.
    """
    # Read the content of the uploaded file into a string (assuming UTF-8 encoding)
    file_content = csv_file.read().decode('utf-8')
    
    # Strip the BOM if present
    if file_content.startswith('\ufeff'):
        file_content = file_content[1:]

    # Use StringIO to simulate a text file object
    string_io_obj = StringIO(file_content)
    
    # Now use csv.DictReader to read the simulated file object
    reader = csv.DictReader(string_io_obj)
    data = [row for row in reader]

    json_data = json.dumps(data)
    return json_data


def extract_text_from_different_file_types(file) -> str:
    """
    Extract text from a file of various types including PDF, TXT, RTF, and ZIP.
    A file with another extension is treated as a .txt file.
    Args:
        file (file-like object): The file from which to extract text.

    Returns:
        str: The extracted text.
    """
    type = file.name.split('.')[-1].lower()
    if type == 'zip':
        st.session_state.zip_file = True
        text = extract_text_from_zip(file)
    elif type == 'pdf':
        text = extract_text_from_pdf(file)
    elif type in ['txt', 'rtf']:
        raw_text = file.read().decode("utf-8")
        text = rtf_to_text(raw_text) if type == 'rtf' else raw_text
    elif type == 'csv':
        text = extract_jason_from_csv(file) # in fact in json format
    else:  # Treat other file type as .txt file
        text = file.read().decode("utf-8")  # Treat all other types as text files

    return text


def increment_file_uploader_key():
    """
    Add 1 to the current streamlit session state "file_uploader_key".

    The purpose of this function is to show a fresh and new streamlit "file_uploader" widget
    without the previously drop file, if there is one.
    """
    st.session_state["file_uploader_key"] += 1


def set_both_load_and_search_sessions_to_False():
    st.session_state.load_session = False
    st.session_state.search_session = False


def extract_text_from_zip(zip_file) -> list:
    """
    Unzip a .zip file and sends all content to an LLM AP.

    Args:
        zip_file: The uploaded zip file.
    Returns:
        str: The extracted text.
    """
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Open the ZIP file in read mode
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract all files to the current working directory
            zip_ref.extractall(temp_dir)

        # Initialize an empty list to store the extracted files with their contents
        extracted_files = []

        # Loop through each file in the ZIP archive
        for filename in zip_ref.namelist():
            # Check if the filename start with '__MACOSX' and skip it
            if not filename.startswith('__MACOSX'):
                # Construct the full path of the extracted file
                full_path = os.path.join(temp_dir, filename)
                # Check if the path is a directory
                if not os.path.isdir(full_path):
                    try:
                        # Open the extracted file in read mode
                        with open(full_path, 'r') as file:
                            # Read the content of the file
                            content = file.read()
                            # Append the file name and content to the list
                            extracted_files.append((filename, content))
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

    return extracted_files


def change_to_prompt_text(extracted_text, _question) -> str:
    """
    Convert the extracted files and their contents into a prompt text for an LLM API.
    """

    # Loop through each file and its content
    if st.session_state.zip_file:
        # Initialize an empty string to store the prompt text
        prompt_text = ""
        for filename, content in extracted_text:
            # Append the file name and content to the prompt text
            prompt_text += f"File: \n{filename}\n\nContent: \n{content}\n\n"
            # Add a separator between files
            prompt_text += "-----\n"
        st.session_state.zip_file = False
    else:
        prompt_text = extracted_text

    # prompt_text =f"```\n{prompt_text}\n```" 

    llm_prompt = (
        "You are a helpful assistant.\n"
        "Context information from a file (or files) and their contents is below.\n"
        "---------------------\n"
        f"{prompt_text}\n"
        "---------------------\n"
        "Given the information above answer the query below.\n"   
        "Your answer should provide the main insights and patterns that can be \n"
        "derived from the files. Do not answer in a code block.\n"
        f"Query: {_question}\n"
        "Answer: "
    )
    # Wrap the text in triple backticks as code text to prevent "#" be interpreted as header in markdown.
    llm_prompt =f"```\n{llm_prompt}\n```"   
    
    # Return the prompt text
    return llm_prompt
