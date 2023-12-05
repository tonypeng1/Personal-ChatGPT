import openai
import streamlit as st
import mysql.connector
from datetime import datetime
from datetime import timedelta
import textwrap
# from streamlit_modal import Modal

def init_connection():
    try:
        conn = mysql.connector.connect(**st.secrets["mysql"])

        with conn.cursor() as cursor:
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session
                (
                    session_id INT AUTO_INCREMENT PRIMARY KEY,
                    start_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_timestamp DATETIME,
                    summary TEXT
                );
            """
            )
            cursor.execute(
            """CREATE TABLE IF NOT EXISTS message
                (
                    message_id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    role TEXT,
                    content TEXT,
                    FOREIGN KEY (session_id) REFERENCES session(session_id)
                );
            """
            )
        conn.commit()

    except mysql.connector.Error as error:
        st.write(f"Failed to create tables: {error}")

    return conn

def load_previous_chat_session(session1):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            st.session_state.messages = []
            for (role, content) in cursor:
                st.session_state.messages.append({"role": role, "content": content})

    except mysql.connector.Error as error:
        st.write(f"Failed to load previous chat sessions: {error}")

# def return_format_function(level1):
#     if level1 == "Today":

# def get_session_id_by_summary(summary1):
#     try:
#         with conn.cursor() as cursor:
#             sql = "SELECT session_id FROM session WHERE summary = %s"
#             val = (summary1,)
#             cursor.execute(sql, val)

#             id = cursor.fetchone()
#             if id:
#                 return id[0]
#             return None

#     except mysql.connector.Error as error:
#         st.write(f"Fail to get session id by summary: {error}")

def load_previous_chat_session_first_question_for_summary(session1):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s LIMIT 1"
            val = (session1,)
            cursor.execute(sql, val)
            row = cursor.fetchone()
            if row and row[0] == "user":
                return textwrap.shorten(row[1], 150, placeholder=' ~')
            return None

    except mysql.connector.Error as error:
        st.write(f"Failed to load previous chat session for summary: {error}")
        return None

def load_previous_chat_session_all_questions_for_summary(session1):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            chat = str()
            for (role, content) in cursor:
                chat += (role + ": " + content + " ")
            return chat

    except mysql.connector.Error as error:
        st.write(f"Failed to load previous chat sessions for summary: {error}")

def load_previous_chat_session_all_questions_for_summary_only_users(session1):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT role, content FROM message WHERE session_id = %s"
            val = (session1,)
            cursor.execute(sql, val)

            chat_user = str()
            for (role, content) in cursor:
                if role == 'user':
                    chat_user += (role + ": " + content + " ")
            return chat_user

    except mysql.connector.Error as error:
        st.write(f"Failed to load previous chat sessions for summary (user only): {error}")

def load_previous_chat_session_id(date_start, date_end):
    try:
        with conn.cursor() as cursor:
            sql = "SELECT DISTINCT(session_id) AS d FROM message WHERE DATE(timestamp) BETWEEN %s AND %s ORDER BY d DESC"
            val = (date_start, date_end)
            cursor.execute(sql, val)

            result = []
            for session in cursor:
                result.append(session[0])

            if not bool(result):
                result = [0]

            return result

    except mysql.connector.Error as error:
        st.write(f"Failed to load chat session id: {error}")

# def get_new_session_id():
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute(
#             """
#             SELECT MAX(session_id) FROM session;
#             """
#             )
#             result = cursor.fetchone()
#             if result is not None and result[0] is not None:
#                 st.session_state.session = result[0] + 1
#             else:
#                 st.session_state.session = 1

#     except mysql.connector.Error as error:
#         st.write(f"Failed to get new session id: {error}")

#     st.write(f"Within the get_new_session_id app, the session is: {st.session_state.session}")

def get_and_set_current_session_id():
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
                st.session_state.session = 1

    except mysql.connector.Error as error:
        st.write(f"Failed to get the current session id: {error}")

    # st.write(f"Within the get_and_set_current_session_id function, the session is: {st.session_state.session}")

def determine_if_the_current_session_is_not_closed():
    try:
        with conn.cursor() as cursor:
            # st.write(f"Current session is {st.session_state.session}")
            sql = "SELECT end_timestamp FROM session WHERE session_id = %s"
            val = (st.session_state.session,)
            cursor.execute(sql, val)

            end = cursor.fetchone()
            # st.write(f"end stamp returns: {end}")
            if end is None:
                return True
            elif end[0] is None:
                return True
            else:
                return False

    except mysql.connector.Error as error:
        st.write(f"Failed to determine if the current session is closed: {error}")

def start_session_save_to_mysql():
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            INSERT INTO session (end_timestamp) VALUE (null);
            """
            )
            conn.commit()

            st.session_state.session += 1

    except mysql.connector.Error as error:
        st.write(f"Failed to start a new session: {error}")

def end_session_save_to_mysql():
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET end_timestamp = %s WHERE session_id = %s;"
            val = (time, st.session_state.session)
            cursor.execute(sql, val)
            conn.commit()

    except mysql.connector.Error as error:
        st.write(f"Failed to end a session: {error}")

    get_session_summary_and_save_to_session_table(st.session_state.session)  # Save session summary after ending session

def save_session_summary_to_mysql(id, summary_text):
    try:
        with conn.cursor() as cursor:
            sql = "UPDATE session SET summary = %s WHERE session_id = %s;"
            val = (summary_text, id)
            cursor.execute(sql, val)
            conn.commit()

    except mysql.connector.Error as error:
        st.write(f"Failed to save the summary of a session: {error}")

def save_to_mysql_message(session_id1, role1, content1):
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO message (session_id, role, content) VALUES (%s, %s, %s)"
            val = (session_id1, role1, content1)
            cursor.execute(sql, val)
            conn.commit()

    except mysql.connector.Error as error:
        st.write(f"Failed to save new message: {error}")

def get_session_summary_and_save_to_session_table(session_id1):
    # chat_session_text = load_previous_chat_session_first_question_for_summary(s1)
    # chat_session_text = load_previous_chat_session_all_questions_for_summary(session_id1)
    chat_session_text_user_only = load_previous_chat_session_all_questions_for_summary_only_users(session_id1)
    # st.write(f"Session text (user only): {chat_session_text_user_only} /n")
    # session_summary = chat_session_text
    session_summary = chatgpt_summary_user_only(chat_session_text_user_only)
    # st.write(f"Summary of session {session_id1}: {session_summary} /n")
    save_session_summary_to_mysql(session_id1, session_summary)

def delete_all_rows():
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET FOREIGN_KEY_CHECKS=0;")

            cursor.execute("TRUNCATE TABLE message;")
            cursor.execute("TRUNCATE TABLE session;")

            cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
            conn.commit()

            st.session_state.messages = []

    except mysql.connector.Error as error:
        st.write(f"Failed to finish deleting data: {error}")

def convert_date(date1):
    if date1 == "Today":
        return today, today
    elif date1 == "Yesterday":
        return (today - timedelta(days = 1)), (today - timedelta(days = 1))
    elif date1 == "Previous 7 days":
        return (today - timedelta(days = 7)), (today - timedelta(days = 2))
    elif date1 == "Previous 30 days":
        return (today - timedelta(days = 30)), (today - timedelta(days = 8))
    elif date1 == "Older":
        # st.write(f"The date is of the type: {type((today - timedelta(days = 30)))}")
        if date_earlist is not None:
            if date_earlist < (today - timedelta(days = 30)):
                return date_earlist, (today - timedelta(days = 30))
            else:
                return (date_earlist - timedelta(days = 31)), (date_earlist - timedelta(days = 31))
        else:
            return today, today # since there is no data in the tables, just return an arbratriry date

def get_the_earliest_date():
    try:
        with conn.cursor() as cursor:
            sql = "SELECT MIN(DATE(timestamp)) FROM message"
            cursor.execute(sql)

            earliest = cursor.fetchone()
            if earliest:
                return earliest[0]
            else:
                return None

    except mysql.connector.Error as error:
        st.write(f"Failed to get the earliest date: {error}")

def get_summary_by_session_id_return_dic(session_id_list):
    try:
        if session_id_list == [0]:
            summary_dict = {0: "No sessions available"}
            return summary_dict
        else:
            with conn.cursor() as cursor:
                # Use a single query to get all summaries
                format_strings = ','.join(['%s'] * len(session_id_list))
                cursor.execute("SELECT session_id, summary FROM session WHERE session_id IN (%s)" % format_strings,
                                tuple(session_id_list))

                summary_dict = {}
                i = 0
                for session_id, summary in cursor.fetchall():
                    # st.write(f"session_id is: f{session_id} and summary is: {summary}")
                    if summary is not None:
                        summary_dict[session_id] = summary
                        i += 1
                    else:
                        if i == 0:
                            summary_dict[session_id] = "Summary not yet available for the current active session."
                        else:
                            pass
                return dict(reversed(list(summary_dict.items())))

        # summary_list = []
        # with conn.cursor() as cursor:
        #     for s in session_id_list:
        #         sql = "SELECT summary FROM session WHERE session_id = %s"
        #         val = (s,)
        #         cursor.execute(sql, val)

        #         summary = cursor.fetchone()
        #         if summary:
        #             summary_list.append(summary[0])

        #     dic = {session_id_list[i]: summary_list[i] for i in range(len(session_id_list))}

        #     return dic

    except mysql.connector.Error as error:
        st.write(f"Failed to load summary: {error}")
        return None

def chatgpt(prompt1):
    determine_if_terminate_current_ssession_and_start_a_new_one()
    st.session_state.messages.append({"role": "user", "content": prompt1})
    with st.chat_message("user"):
        st.markdown(prompt1)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": "You are based out of Austin, Texas. You are a software engineer " +
                    "predominantly working with Kafka, java, flink, Kafka-connect, ververica-platform. " +
                    "You also work on machine learning projects using python, interested in generative AI and LLMs. " +
                    "You always prefer quick explanations unless specifically asked for. When rendering code samples " +
                    "always include the import statements. When giving required code solutions include complete code " +
                    "with no omission. When giving long responses add the source of the information as URLs. " +
                    "Assume the role of experienced Software Engineer and You are fine with strong opinion as long as " +
                    "the source of the information can be pointed out and always question my understanding. " +
                    "When rephrasing paragraphs, use lightly casual, straight-to-the-point language."}] +
                [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                ],
            temperature=temperature,
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    save_to_mysql_message(st.session_state.session, "user", prompt1)
    save_to_mysql_message(st.session_state.session, "assistant", full_response)

    # get_session_summary_and_save_to_session_table(st.session_state.session)

def chatgpt_summary(chat_text):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Use a few words to describe the main topics of the following chat session. No more than five words.\n\n" +
      chat_text,
      max_tokens=50,  # Adjust the max tokens as per the summarization requirements
      n=1,
      stop=None,
      temperature=0.5
      )
    summary = response.choices[0].text.strip()

    return summary

def chatgpt_summary_user_only(chat_text_user_only):
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Use a sentence to describe the main topics of the user's questions in following chat session. " + 
      "No more than sixteen words in a sentence. Do not start the sentence with 'The user' or 'Questions about'. " + 
      "A partial sentence is fine.\n\n" +
      chat_text_user_only,
      max_tokens=50,  # Adjust the max tokens as per the summarization requirements
      n=1,
      stop=None,
      temperature=0.5
      )
    summary = response.choices[0].text.strip()

    return summary

def determine_if_terminate_current_ssession_and_start_a_new_one():
    # st.write(f"value of st.session_state.session_not_close when enter determine function is: {st.session_state.session_not_close}")
    if st.session_state.first_run and ((not st.session_state.load_history_level_2 and not st.session_state.file_upload)
                                       and not st.session_state.new_session):
        # get_new_session_id()  # Get the session_id after "get_new_session_id()" is run. To be used when saving to the message table.
        start_session_save_to_mysql()  # always run after "get_new_session_id()"
        get_and_set_current_session_id()
        st.session_state.first_run = False
    elif st.session_state.new_session:
        end_session_save_to_mysql()
        start_session_save_to_mysql()
        get_and_set_current_session_id()
        st.session_state.new_session = False
        st.session_state.first_run = False
    elif st.session_state.session_not_close:
        # st.write("Enter session_not_vlose elif is True")
        end_session_save_to_mysql()
        start_session_save_to_mysql()
        get_and_set_current_session_id()

        for message in st.session_state.messages:  # Since a new promp is entered, need to save previous messages to mysql.
            save_to_mysql_message(st.session_state.session, message["role"], message["content"])
            # st.write(f"save to message table: {message['role']}: {message['content']}")

        st.session_state.session_not_close = False
    elif st.session_state.load_history_level_2:
        # st.write("Code enters determine_if_terminate_current_ssession_and_start_a_new_one level 2")
        end_session_save_to_mysql()
        start_session_save_to_mysql()
        get_and_set_current_session_id()

        for message in st.session_state.messages:  # Since a new promp is entered, need to save previous messages to mysql.
            save_to_mysql_message(st.session_state.session, message["role"], message["content"])

        st.session_state.selectbox_first_level_index = None
        st.session_state.load_history_level_2 = False
        st.session_state.first_run = False
    elif st.session_state.file_upload:
        end_session_save_to_mysql()
        start_session_save_to_mysql()
        get_and_set_current_session_id()
        st.session_state.file_upload = False
        st.session_state.first_run = False
    else:
        pass


openai.api_key = st.secrets["OPENAI_API_KEY"]
conn = init_connection()

today = datetime.now().date()
time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
date_earlist = get_the_earliest_date()
# st.write(f"The earlist date is: {date_earlist}")
# st.write(f"The earlist date is of the type: {type(date_earlist)}")

st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
temperature = st.sidebar.number_input("Input the temperture value (from 0 to 1.6):",
                                      min_value=0.0, max_value=1.6, value=1.0, step=0.2)

# Initialize session states
if 'selectbox_first_level_index' not in st.session_state:  # if None, a selection box will initialize empty.
    st.session_state.selectbox_first_level_index = None

if "first_run" not in st.session_state:
    # st.write(f"first run not in st.session_state is True")
    st.session_state.first_run = True  # need to reset if used later in code to start a session.

if "session_not_close" not in st.session_state:
    st.session_state.session_not_close = False

if "session" not in st.session_state:
    # st.write(f"session not in st.session_state is True")
    get_and_set_current_session_id()  # get current session id (i.e., "st.session_state.session") to either 1 or = the max id in table
    st.session_state.session_not_close = determine_if_the_current_session_is_not_closed()
    if st.session_state.session_not_close:
        load_previous_chat_session(st.session_state.session)
        st.session_state.first_run = False

# st.write(f"Current session is: {st.session_state.session}")

if "openai_model" not in st.session_state:
    st.session_state.openai_model = model_name

if "messages" not in st.session_state:
    st.session_state.messages = []

if "new_session" not in st.session_state:
    st.session_state.new_session = False

if "load_history_level_2" not in st.session_state:
    st.session_state.load_history_level_2 = False

if "file_upload" not in st.session_state:
    st.session_state.file_upload = False

if "empty_data" not in st.session_state:
    st.session_state.empty_data = False

if "confirmation" not in st.session_state:
    st.session_state.confirmation = False

load_history_level_1 = st.sidebar.selectbox(
    label='Load a previous chat date:',
    placeholder='Pick a date',
    options=['Today', 'Yesterday', 'Previous 7 days', 'Previous 30 days', 'Older'],
    index=st.session_state.selectbox_first_level_index,
    key="first_level"
    )

# list of session_ids of a date range
today_sessions = load_previous_chat_session_id(*convert_date('Today'))
# st.write(f"today's session ids: {today_sessions}")
yesterday_sessions = load_previous_chat_session_id(*convert_date('Yesterday'))
seven_days_sessions = load_previous_chat_session_id(*convert_date('Previous 7 days'))
thirty_days_sessions = load_previous_chat_session_id(*convert_date('Previous 30 days'))
older_sessions = load_previous_chat_session_id(*convert_date('Older'))

today_dic = get_summary_by_session_id_return_dic(today_sessions)
# st.write(f"today's returned dic: {today_dic}")
yesterday_dic = get_summary_by_session_id_return_dic(yesterday_sessions)
seven_days_dic = get_summary_by_session_id_return_dic(seven_days_sessions)
thirty_days_dic = get_summary_by_session_id_return_dic(thirty_days_sessions)
older_dic = get_summary_by_session_id_return_dic(older_sessions)

level_two_options = {
    None : {0: "No date selection"},
    "Today" : today_dic,
    "Yesterday" : yesterday_dic,
    "Previous 7 days" : seven_days_dic,
    "Previous 30 days" : thirty_days_dic,
    "Older" : older_dic
}

load_history_level_2 = st.sidebar.selectbox(
        label="Load a previous chat session:",
        placeholder='Pick a session',
        options=list((level_two_options[load_history_level_1]).keys()),
        index=None,
        format_func=lambda x:level_two_options[load_history_level_1][x]
        )
# st.write(f"The return of level 2 click is: {load_history_level_2}")

new_chat_button = st.sidebar.button("New chat session", key="new")
empty_database = st.sidebar.button("Delete chat history in database")
uploaded_file = st.sidebar.file_uploader("Upload a file")
to_chatgpt = st.sidebar.button("Send to chatGPT")

if load_history_level_2:
    # st.write(f"session choice: {load_history_level_2}")
    st.session_state.load_history_level_2 = True
    load_previous_chat_session(load_history_level_2)
    # st.write(f"value of st.session_state.load_history_level_2 when first click is: {st.session_state.load_history_level_2}")

if new_chat_button:
    st.session_state.new_session = True  # This state is needed to determin if a new session needs to be created
    # st.write(f"Enter new chat: {new_chat_button}")
    st.session_state.messages = []
    st.session_state.load_history_level_2 = False

# modal = Modal("Attention", key="popup")
# if empty_database:
#     modal.open()
#     if modal.is_open():
#         with modal.container():
#             st.markdown("Do you really, realy, wanna do this ?")
#             yes = st.button("Yes")
#             no = st.button("No")
    # st.error("Do you really, really, wanna do this?", icon="🚨")
    # delete_all_rows()
    # st.session_state.messages = []
    # st.sidebar.write("Chat history in database is empty!")

    # get_new_session_id()
    # start_session_save_to_mysql()

if empty_database:
    st.session_state.empty_data = True
    st.error("Do you really, really, wanna delete all chat history?", icon="🚨")

if st.session_state.empty_data:
    confirmation = st.selectbox(
        label="Confirm your answer (If you choose 'Yes', ALL CHAT HISTORY in the database will be deleted):",
        placeholder="Pick a choice",
        options=['No', 'Yes'],
        index=None
    )
    if confirmation == 'Yes':
        delete_all_rows()
        st.warning("Data deleted.", icon="🚨")
        st.session_state.empty_data = False
        st.session_state.new_session = True
    elif confirmation == 'No':
        st.success("Data not deleted.")
        st.session_state.empty_data = False
    # confirmation = st.checkbox("Check to confirm")
    # if confirmation:
    #     st.warning("Data deleted.", icon="🚨")
    # st.session_state.empty_data = False

# if empty_database:
#     st.error("Do you really, really, wanna do this?", icon="🚨")
#     st.text_input("Type 'yes' if you really want to do this.", value="No", key='choice')
#     if st.session_state.choice == 'yes':
#         st.write("data deleted")
#     else:
#         st.write("data not deleted")

# Print each message on page
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file is not None and not to_chatgpt:
    prompt_f = uploaded_file.read().decode("utf-8")
    st.sidebar.write(prompt_f)

# Call openai api
if uploaded_file is not None and to_chatgpt:
    prompt_f = uploaded_file.read().decode("utf-8")
    st.session_state.file_upload = True
    chatgpt(prompt_f)

if prompt := st.chat_input("What is up?"):
    chatgpt(prompt)

conn.close()