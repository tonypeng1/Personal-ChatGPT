import streamlit as st
from mysql.connector import connect, Error


def insert_initial_default_model_type(conn, type1: str) -> None:
    """
    Inserts the initial default model type into the 'model' table if it does not already exist.

    This function attempts to insert a new row into the 'model' table with the provided choice.
    The insertion will only occur if the table is currently empty, ensuring that the default
    type is set only once.

    Args:
        conn: A connection object to the database.
        type1: A string representing the default type to be inserted.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO model (choice) 
            SELECT %s FROM DUAL 
            WHERE NOT EXISTS (SELECT * FROM model);
            """
            val = (type1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save the initial default model type: {error}")
        raise


def Load_the_last_saved_model_type(conn) -> None:
    """
    Retrieves the last saved model type from the 'model' table in the MySQL database and
    save it to session state.

    This function selects the most recent 'choice' entry from the 'mdoel' table based on the highest ID.

    Args:
        conn: A connection object to the MySQL database.

    Returns:
        The last saved model type as a string, or None if no type is found.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT choice FROM model ORDER BY id DESC LIMIT 1;")

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                if result[0] in ("perplexity-llama-3-sonar-large-32k-chat", 
                                 "perplexity-llama-3-sonar-large-32k-online",
                                 "perplexity-llama-3.1-sonar-large-128k-online",
                                 "perplexity-llama-3.1-sonar-huge-128k-online"):  # for v0.7.0 and v0.8.0
                    result = ("perplexity-sonar-pro", )
                if result[0] in ("gemini-1.5-pro-latest",
                                 "gemini-1.5-pro-exp-0801",
                                 "gemini-1.5-pro-002",
                                 "gemini-exp-1121",
                                 "gemini-2.0-flash-exp",
                                 "gemini-2.0-flash"
                                 ):
                    result = ("gemini-3-pro-preview", )
                if result[0] in ("gemini-2.0-flash-thinking-exp", 
                                 "gemini-2.0-flash-thinking-exp-01-21",
                                 "gemini-2.5-pro-preview-03-25",
                                 "gemini-2.5-pro-preview-05-06",
                                 "gemini-2.5-pro"
                                 ):
                    result = ("gemini-3-pro-preview-thinking", )
                if result[0] in ("claude-3-opus-20240229", 
                                 "claude-3-5-sonnet-20240620",
                                 "claude-3-5-sonnet-20241022",
                                 "claude-3-7-sonnet-20250219",
                                 "claude-sonnet-4-20250514"):
                    result = ("claude-sonnet-4-5-20250929", )
                if result[0] in ("claude-3-7-sonnet-20250219-thinking", 
                                 "claude-sonnet-4-20250514-thinking"):
                    result = ("claude-sonnet-4-5-20250929-thinking", )
                if result[0] in ("gpt-4-turbo-2024-04-09", 
                                 "gpt-4o-2024-11-20",
                                 "o1-preview",
                                 "gpt-4o",
                                 "gpt-4.1-2025-04-14",
                                 "gpt-5-2025-08-07",
                                 "gpt-5-mini-2025-08-07",
                                 "gpt-5.1-2025-11-13"):
                    result = ("gpt-5.2-2025-12-11", )
                if result[0] in ("mistral-large-latest", ):
                    result = ("pixtral-large-latest", )
                if result[0] in ("Qwen2.5-Coder-32B-Instruct",
                                 "Qwen2.5-Max",
                                 "Qwen3-235b-a22b"):
                    result = ("qwen3-235b-a22b-2507", )
                if result[0] in ("DeepSeek-R1", ):
                    result = ("DeepSeek-R1-0528", )
                if result[0] in ("o3-mini-high", 
                                 "gpt-5-mini-2025-08-07-thinking",
                                 "gpt-5.1-2025-11-13-thinking"):
                    result = ("gpt-5.2-2025-12-11-thinking", )
                st.session_state.type = result[0]
            else:
                st.session_state.type = None

    except Error as error:
        st.error(f"Failed to read the last saved type: {error}")
        raise


def return_type_index(type1: str) -> int:
    """
    Returns the index of a given type from a predefined dictionary of types.

    This function maps a type description to its corresponding index based on a predefined
    dictionary of type descriptions and their associated indices.

    Args:
        type (str): A string representing the type description.

    Returns:
        An integer representing the index of the type.

    Raises:
        KeyError: If the type description is not found in the predefined dictionary.
    """
    type_dic = {
        "gpt-5.2-2025-12-11": 0,
        "gpt-5.2-2025-12-11-thinking": 1,
        "claude-sonnet-4-5-20250929": 2, 
        "claude-sonnet-4-5-20250929-thinking": 3, 
        "pixtral-large-latest": 4,
        "gemini-3-pro-preview":5,
        "gemini-3-pro-preview-thinking":6,
        "DeepSeek-R1-0528": 7,
        "perplexity-sonar-pro": 8,
        "nvidia-llama-3.1-nemotron-70b-instruct": 9,
        "qwen3-235b-a22b-2507": 10
    }
    if type1 not in type_dic:
        raise KeyError(f"Type '{type1}' not found in the type dictionary.")
    
    return type_dic[type1]


def save_model_type_to_mysql(conn, type1: str) -> None:
    """
    Saves a model type to the 'model' table in the MySQL database.

    This function inserts a new row into the 'model' table with the provided choice.

    Args:
        conn: A connection object to the MySQL database.
        type1: A string representing the type to be saved.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO model (choice) VALUE (%s)"
            val = (type1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save model type: {error}")
        raise


