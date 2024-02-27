from mysql.connector import Error
import streamlit as st


def delete_the_messages_of_a_chat_session(conn, session_id1: int) -> None:
    """
    Deletes all messages associated with a specific chat session from the database.

    This function executes a DELETE SQL command to remove messages from the 'message' table
    where the 'session_id' matches the provided session ID.

    Args:
        conn: A connection object to the MySQL database.
        session_id (int): The ID of the chat session whose messages are to be deleted.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "DELETE FROM message WHERE session_id = %s"
            val = (session_id1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to delete the messages of a chat session: {error}")
        raise


def delete_all_rows(conn) -> None:
    """Delete all rows from 'message' and 'session' tables.

    This function will disable foreign key checks, truncate the message and
    session tables, and then re-enable foreign key checks to maintain integrity.

    Parameters:
    - conn: MySQLConnection object representing the database connection.

    Raises:
    - Raises an exception if any SQL errors occur during the deletion process.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
            cursor.execute("TRUNCATE TABLE message;")
            cursor.execute("TRUNCATE TABLE session;")
            cursor.execute("TRUNCATE TABLE behavior;")
            cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
            conn.commit()

            st.session_state.messages = []

    except Error as error:
        st.error(f"Failed to finish deleting data: {error}")
        raise