from datetime import datetime
from typing import Tuple, List

from mysql.connector import Error
import streamlit as st


def delete_all_rows_in_message_serach(conn) -> None:
    try:
        with conn.cursor() as cursor:
            cursor.execute("TRUNCATE TABLE message_search;")
            conn.commit()

    except Error as error:
        st.error(f"Failed to finish deleting data in message_search table: {error}")
        raise


def search_keyword_and_save_to_message_search_table(conn, words: str):
    """
    Searches for messages containing the given keywords and saves the results to the message_search table.

    Parameters:
    conn (MySQLConnection): A connection object to the MySQL database.
    words (str): A string containing keywords separated by spaces.

    Raises:
    Raises an exception if the search or saving to the message_search table fails.
    """
    try:
        with conn.cursor() as cursor:
            keywords1 = words.split()
            sql_conditions, filtered_word_list = filter_word_list_and_get_sql_conditions(keywords1)
            
            sql = f"SELECT * FROM message WHERE {sql_conditions}"
            val = tuple(f"%{keyword}%" for keyword in filtered_word_list)
            cursor.execute(sql, val)

            for mess_id, sess_id, time, user, model, content in cursor.fetchall():
                save_to_mysql_message_search(conn, mess_id, sess_id, time, user, model, \
                                             content)

    except Error as error:
        st.error(f"Failed to search keyword: {error}")
        raise


def save_to_mysql_message_search(conn, message_id1: int, session_id1: int, 
                                 timestamp1: datetime, role1: str, model1: str, 
                                 content1: str) -> None:
    """
    Inserts a new message into the message_search table in the MySQL database.

    Parameters:
    conn (MySQLConnection): A connection object to the MySQL database.
    message_id1 (int): The ID of the message.
    session_id1 (int): The ID of the session.
    timestamp1 (datetime): The timestamp when the message was sent.
    role1 (str): The role of the user who sent the message.
    model1 (str): The mdoel used.
    content1 (str): The content of the message.

    Returns:
    None
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO message_search 
            (message_id, session_id, timestamp, role, model, content) 
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            val = (message_id1, session_id1, timestamp1, role1, model1, content1)
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save new message to message_search table: {error}")
        raise


def filter_word_list_and_get_sql_conditions(word_list: List[str]) -> Tuple[str, List[str]]:
    """
    Filters out 'or' and 'and' from the word list and creates a SQL condition string.

    Parameters:
    word_list (List[str]): A list of words to filter and use for creating the SQL condition.

    Returns:
    Tuple[str, List[str]]: A tuple containing the SQL condition string and the filtered word list.
    """
    filtered_word_list = [word for word in word_list 
                          if word.lower() not in ("or", "and")]
    contains_or = any(word.lower() == "or" for word in word_list)

    condition_operator = " OR " if contains_or else " AND "
    conditions = condition_operator.join(["content LIKE %s" for _ in filtered_word_list])
    return (conditions, filtered_word_list)


def search_full_text_and_save_to_message_search_table(conn, words: str):
    """
    Full-text searches for messages and saves the results to the message_search table.

    Parameters:
    conn (MySQLConnection): A connection object to the MySQL database.
    words (str): A string containing search words separated by spaces.

    Raises:
    Raises an exception if the search or saving to the message_search table fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT * FROM message 
            WHERE MATCH (content) AGAINST 
            (%s IN BOOLEAN MODE)
            """
            val = (words,)
            cursor.execute(sql, val)

            for mess_id, sess_id, time, user, model, content in cursor.fetchall():
                save_to_mysql_message_search(conn, mess_id, sess_id, time, user, model, \
                                             content)

    except Error as error:
        st.error(f"Failed to search full-text: {error}")
        raise