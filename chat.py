import openai
import streamlit as st
import mysql.connector
from datetime import date

def init_connection():
    conn = mysql.connector.connect(**st.secrets["mysql"])

    with conn.cursor() as cursor:
        cursor.execute(
        """CREATE TABLE IF NOT EXISTS history
            (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Date DATE,
                Role TEXT,
                Content TEXT
            );"""
        )

    return conn

def load_previous_chat():
    with conn.cursor() as cursor:
        sql = "SELECT Date, Role, Content FROM history"
        cursor.execute(sql)

        st.session_state.messages = []
        for (date, role, content) in cursor:
            st.session_state.messages.append({"role": role, "content": content})

def save_to_mysql(role1, content1):
    with conn.cursor() as cursor:
        sql = "INSERT INTO history (Date, Role, Content) VALUES (%s, %s, %s)"
        val = (today, role1, content1)
        cursor.execute(sql, val)
        conn.commit()

def delete_all_rows():
    with conn.cursor() as cursor:
        sql = "DELETE FROM history"
        cursor.execute(sql)
        conn.commit()

def chatgpt(prompt1):
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
    save_to_mysql("user", prompt1)
    save_to_mysql("assistant", full_response)

openai.api_key = st.secrets["OPENAI_API_KEY"]
today = date.today()
conn = init_connection()

st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
temperature = st.sidebar.number_input("Input the temperture value (from 0 to 1.6):",
                                      min_value=0.0, max_value=1.6, value=1.0, step=0.2)
load_history = st.sidebar.button("Load chat history")
new_chat_button = st.sidebar.button("New chat", key="new")
empty_database = st.sidebar.button("Delete chat history in database")
uploaded_file = st.sidebar.file_uploader("Upload a file")
to_chatgpt = st.sidebar.button("Send to chatGPT")


if "first_run" not in st.session_state or st.session_state.first_run:
    if load_history:
        load_previous_chat()
        st.session_state.first_run = False

if new_chat_button:
    st.session_state.messages = []
    st.session_state.first_run = True
    # delete_all_rows()

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model_name

if empty_database:
    delete_all_rows()
    st.session_state.messages = []
    st.sidebar.write("Chat history in database is empty!")

# Create or reset "messages" in session state as an empty list
if "messages" not in st.session_state:
    st.session_state.messages = []

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
    chatgpt(prompt_f)

if prompt := st.chat_input("What is up?"):
    chatgpt(prompt)

conn.close()