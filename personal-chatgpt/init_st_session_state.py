import streamlit as st


def init_session_states():
    """
    Initialize the session states used in the Streamlit app when the app is first run or is
    reloaded (by the user clicking on the "Reload this page" button at the browser).
    
    1. Four session states are used for different database operations before sending an
    user input to a LLM. Either there is only one session state in this group is true or
    all of the session states are False (the case where the current active session continues).
    - `new_table`: A boolean indicating whether the `session` table in database is new (empty).
    - `new_session`: A boolean indicating whether the current session should be closed and a 
                new session be created.
    - `load_history_level_2`: A boolean indicating whether the messages of a previous 
                            chat session have been loaded.
    - `session_different_date`: A boolean indicating whether today is different from the date
                            at which the current active session was first created.

    2. Two session states related to message deletion:
    - `delete_session`: A boolean indicating whether the user wants to delete the messages of 
                a session from database.
    - `empty_data`: A boolean indicating whether the user wants to empty All data in ALL tables 
                    in the database.
    
    3. Three session states related to dropping a file (or files) to LLM:
    - `send_drop_file`: A boolean indicating whether the user wants to drop a file (or files) 
                    from local computer to the LLM.
    - `question`: A boolean indicating whether a question has been asked about a dropped file
                (or files).
    - `file_uploader_key`: An integer representing the key for the streamlit file_uploader widget.
                    This key is incremented and is used to create a new file_uploader widget 
                    without showing a previously loaded file (or files, if there is any).

    4. Miscellaneous:
    - `behavior`: A string representing the current behavior of the LLM.
    - `messages`: A list of chat messages with roles and contents of a chat session to be 
                displayed to the user.
    - `load_session`: A boolean indicating whether the user wants to load the messages of a 
                    previous session.
    - `search_session`: A boolean indicating whether the user wants to search the messages of a 
                    previous session.
    """

    if "behavior" not in st.session_state:
        st.session_state.behavior = ""

    if "new_table" not in st.session_state:
        st.session_state.new_table = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "new_session" not in st.session_state:
        st.session_state.new_session = False

    if "load_history_level_2" not in st.session_state:
        st.session_state.load_history_level_2 = 0  # session ID that starts from 1

    if "delete_session" not in st.session_state:
        st.session_state.delete_session = False

    if "empty_data" not in st.session_state:
        st.session_state.empty_data = False

    if "question" not in st.session_state: # If a question has been asked about a file, bypass the text_area()
        st.session_state.question = False

    if "session_different_date" not in st.session_state:
        st.session_state.session_different_date = False

    if "load_session" not in st.session_state:
        st.session_state.load_session = False

    if "search_session" not in st.session_state:
        st.session_state.search_session = False

    if "send_drop_file" not in st.session_state:
        st.session_state.send_drop_file = False

    if "file_uploader_key" not in st.session_state:
        st.session_state.file_uploader_key = 0