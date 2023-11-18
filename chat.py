import openai
import streamlit as st

def chatgpt():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

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

st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
temperature = st.sidebar.number_input("Input the temperture value (from 0 to 2):",
                                      min_value=0.0, max_value=1.6, value=1.0, step=0.2)
clear_button = st.sidebar.button("Clear Conversation", key="clear")
uploaded_file = st.sidebar.file_uploader("Upload a file")
to_chatgpt = st.sidebar.button("Send to chatGPT")

if uploaded_file is not None and not to_chatgpt:
    prompt = uploaded_file.read().decode("utf-8")
    st.sidebar.write(prompt)

if to_chatgpt:
    prompt = uploaded_file.read().decode("utf-8")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = model_name

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if uploaded_file is not None and to_chatgpt:
    st.sidebar.write(prompt)
    chatgpt()

if prompt := st.chat_input("What is up?"):
    chatgpt()