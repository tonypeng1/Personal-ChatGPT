import streamlit as st
from mysql.connector import connect, Error


def get_and_set_current_session_id(conn) -> None:
    """
    Retrieves the highest session ID from the 'session' table and sets it
    as the current session ID in the Streamlit session state. If no session id 
    is found, set to None.

    Parameters:
    conn (Connection): A mysql connection object to the database.

    Returns:
    None
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            SELECT MAX(session_id) FROM session;
            """
            )
            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                st.session_state.session = result[0]
            else:
                st.session_state.session = None

    except Error as error:
        st.error(f"Failed to get the current session id: {error}")
        raise

def load_previous_chat_session(conn, session1: int) -> None:
    """
    Load messages of a previous chat session from database and append to the Streamlit session state
    "messages".

    Args:
        conn: A MySQL connection object.
        session1: The ID of the chat session to retrieve messages from.

    Returns:
        None. Messages are loaded into `st.session_state.messages`.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, image, model, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            st.session_state.messages = []
            for (role, image, model, content) in cursor:
                st.session_state.messages.append({"role": role, "image": image, "model": model, 
                                                  "content": content})

    except Error as error:
        st.error(f"Failed to load previous chat sessions: {error}")
        raise

def set_only_current_session_state_to_true(current_state: str) -> None:
    """
    Update the session state by setting the specified current state to True and all other states to False.

    This function iterates over a predefined list of states and updates the session state such that only the
    state matching `current_state` is set to True, while all others are set to False.

    Parameters:
    current_state (str): The key in the session state dictionary that should be set to True.

    Returns:
    None
    """
    for state in ["new_table", 
                  "new_session", 
                  "load_history_level_2", 
                  "session_different_date"]:
        st.session_state[state] = (state == current_state)
    
    # DO NOT CHANGE TO THE FOLLOWING CODE. IT WILL BREAK THE "load_history_level_2" value!
    # for state in [
    #     "drop_clip",
    #     "drop_clip_loaded",
    #     "drop_file",
    #     "load_history_level_2", 
    #     "load_session",
    #     "new_table", 
    #     "new_session", 
    #     "search_session",
    #     "session_different_date",
    #     ]:
    #     st.session_state[state] = (state == current_state)


# if __name__ == "__main__":
#     # This code displays the messages of the current active session (most recent session) 
#     # in the database.
    
#     connection = connect(**st.secrets["mysql"])  # Get LOCAL database credentials from .streamlit/secrets.toml for development.

#     if "new_table" not in st.session_state:
#         st.session_state.new_table = False

#     if "session" not in st.session_state:
#         get_and_set_current_session_id(connection)

#         if st.session_state.session is not None:
#             load_previous_chat_session(connection, st.session_state.session)
#         else:
#             set_only_current_session_state_to_true("new_table")

#     # Print meassages on page
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     connection.close()