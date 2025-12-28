import base64

from mistralai import Mistral
from mysql.connector import Error
import ocrspace
from openai import OpenAIError, APITimeoutError, APIConnectionError
import streamlit as st
import tiktoken


def load_previous_chat_session_all_questions_for_summary_only_users_image(conn, session1: str) -> str:
    """
    Loads and concatenates the content of all messages sent by the user in a given chat session.

    Args:
        conn: A MySQL database connection object.
        session_id: The unique identifier for the chat session.

    Returns:
        A string containing all user messages concatenated together, or None if an error occurs.

    Raises:
        Raises an error and logs it with Streamlit if the database operation fails.
    """

    # Set a timeout duration in seconds
    TIMEOUT_DURATION = 5
    chat_user = ""

    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, image, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            for (role, image, content) in cursor.fetchall():  # Fetch all rows at once
                if role == 'user':
                    chat_user += (content or "") + " "
                    if image != "":
                        # Use the OCR API to extract text from the image
                        try:
                            extracted_text = mistral_ocr_api_call(image)
                            chat_user += " " + (extracted_text or "") + " "
                        except Exception as e:
                            st.error(f"Error occurred while processing the image in Mistral OCR API: {e}")
                        
            # Shorten the prompt to 3800 tokens or less
            # print(f"chat_user: {chat_user}")
            chat_user = shorten_prompt_to_tokens(chat_user)
            return chat_user

    except Error as error:
        st.error(f"Failed to load previous chat sessions for summary (user only): {error}")
        raise


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def mistral_ocr_api_call(
        image_file_path,
        ) -> str:
    
    # Set mastral api configuration
    MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
    mistral_model = "pixtral-large-latest"
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)

    ocr_model_role = (
        "You are an expert Optical Character Recognition (OCR) system. Your primary function \n" 
        "is to accurately extract text from images provided by the user. You will receive images \n"
        "as input and must return the text content present in the image. Focus on accuracy and \n"
        "completeness. If the image contains multiple text blocks, extract all of them. If the image \n" 
        "contains no text, respond with an empty string. \n"
    )
    ocr_prompt = (
        "Extract the text elements from the picture. DO NOT REPLY WITH ANY ADDITIONAL WORDS."
    )

    system_list = [{"role": "system", "content": ocr_model_role}]

    base64_image = encode_image(image_file_path)
    context_list = []
    context_list = [{"role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ocr_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_image}",
                        },
                    ],
                }]
    input_list = system_list + context_list

    full_response = ""
    try:
        for chunk in mistral_client.chat.stream(
            model=mistral_model,
            messages=input_list,
            temperature=0.0,
            max_tokens=3000,
        ):
            delta = getattr(chunk.data.choices[0], "delta", None)
            delta_content = getattr(delta, "content", None) if delta else None

            chunk_text = ""
            if isinstance(delta_content, str):
                chunk_text = delta_content
            elif isinstance(delta_content, list):
                text_parts = []
                for item in delta_content:
                    if isinstance(item, str):
                        text_parts.append(item)
                        continue
                    item_text = getattr(item, "text", None)
                    if not item_text and isinstance(item, dict):
                        item_text = item.get("text")
                    if item_text:
                        text_parts.append(item_text)
                chunk_text = "".join(text_parts)

            if chunk_text:
                full_response += chunk_text

    except Exception as e:
        error_response = f"An unexpected error occurred in Mistral OCR API call: {e}"
        full_response = error_response
        st.markdown(full_response)

    return full_response


# def ocr_api_call(image):

#     OCR_API_KEY = st.secrets["OCR_API_KEY"]
#     ocr_api = ocrspace.API(
#         api_key=OCR_API_KEY,
#         OCREngine=2,
#     )

#     extracted_text = ""
#     try:
#         extracted_text = ocr_api.ocr_file(image)
#     except Exception as e:
#         st.error(f"Error occurred while processing the image in ocr API: {e}")
#         return extracted_text

#     return extracted_text


def shorten_prompt_to_tokens(prompt: str, encoding_name: str="cl100k_base" , max_tokens: int=2000) -> str:
    """
    Shortens the input prompt to a specified maximum number of tokens using the specified encoding.
    If the number of tokens in the prompt exceeds the max_tokens limit, it truncates the prompt.

    Parameters:
    - prompt (str): The input text to be potentially shortened.
    - encoding_name (str): The name of the encoding to use for tokenization (default: "cl100k_base").
    - max_tokens (int): The maximum number of tokens that the prompt should contain (default: 3800).

    Returns:
    - str: The original prompt if it's within the max_tokens limit, otherwise a truncated version of it.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    encoding_list = encoding.encode(prompt)
    num_tokens = len(encoding_list)

    if num_tokens > max_tokens:
        truncated_tokens = encoding_list[:max_tokens]
        truncated_prompt = encoding.decode(truncated_tokens)
        return truncated_prompt
    else:
        return prompt


def chatgpt_summary_user_only(client, chat_text_user_only: str) -> str:
    """
    Generates a summary sentence for the main topics of a user's chat input using OpenAI's Completion API.

    Parameters:
    chat_text_user_only (str): The chat text input from the user which needs to be summarized.

    Returns:
    str: A summary sentence of the user's chat input.

    Note:
    The prompt instructs the AI to avoid starting the sentence with "The user" or "Questions about"
    and limits the summary to a maximum of sixteen words.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes chat sessions."},
                {"role": "user", "content": "Use a sentence to summary the main topics of the user's questions in following chat session. " + 
                "DO NOT start the sentence with 'The user' or 'Questions about'. For example, if the summary is 'Questions about handling errors " + 
                "in OpenAI API.', just return 'Handling errors in OpenAI API'. DO NOT use special characters that can not be used in a file name. " + 
                "No more than ten words in the sentence. A partial sentence is fine.\n\n" + chat_text_user_only}
            ],
            max_tokens=30,
            n=1,
            stop=None,
            temperature=0.5,
            timeout=30
        )
        summary = response.choices[0].message.content.strip()
        return summary

    except APIConnectionError as e:
        st.error("Could not connect to OpenAI API. Please check your internet connection or try again later.")
        raise
    except OpenAIError as e:
        st.error(f"An error occurred with OpenAI in getting chat summary: {e}")
        raise
    except APITimeoutError as e:
        st.error("OpenAI API request timed out. Please try again later.")
        raise
    except Exception as e:
        st.error(f"An unexpected error occurred in OpenAI API call to get summary: {e}")
        raise


def save_session_summary_to_mysql(conn, id: int, summary_text: str) -> None:
    """
    Updates the session summary text in the "session" table in a MySQL database.

    Parameters:
    - conn (MySQLConnection): An established MySQL connection object.
    - id (int): The unique identifier for the session to be updated.
    - summary_text (str): The new summary text for the session.

    Raises:
    - Error: If the database update operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET summary = %s WHERE session_id = %s;"
            val = (summary_text, id)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save the summary of a session: {error}")
        raise


def get_session_summary_and_save_to_session_table(conn, client, session_id1: int) -> None:
    """
    Retrieves the chat session's text, generates a summary for user messages only,
    and saves the summary to the session table in the database.

    Parameters:
    - session_id1 (str): The unique identifier of the chat session.

    """
    chat_session_text_user_only = load_previous_chat_session_all_questions_for_summary_only_users_image(conn, session_id1)
    try:
        session_summary = chatgpt_summary_user_only(client, chat_session_text_user_only)
    except Error as error:
        st.error(f"Failed to get summary from openai model: {error}")
        raise
    save_session_summary_to_mysql(conn, session_id1, session_summary)