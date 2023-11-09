# import streamlit as st
# import numpy as np

# with st.chat_message("user"):
#     st.write("Hello! 👋")

# with st.chat_message("assistant"):
#     st.write("Hello human")
#     st.bar_chart(np.random.randn(30, 3))

# message = st.chat_message("assistant")
# message.write("Hello human")
# message.bar_chart(np.random.randn(30, 3))



# import streamlit as st

# prompt = st.chat_input("Say something")

# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")



# import streamlit as st

# st.title('Echo bot')

# if 'messages' not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message['role']):
#         st.markdown(message['content'])

# if prompt := st.chat_input("What's up?"):
#     with st.chat_message('user'):
#         st.markdown(prompt)
#     st.session_state.messages.append({'role': 'user', 'content': prompt})

#     response = f"Echo: {prompt}"
#     with st.chat_message('assistant'):
#         st.markdown(response)

#     st.session_state.messages.append({'role': 'assistant', 'content': response})



# import streamlit as st
# import random
# import time

# st.title("Simple chat")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Accept user input
# if prompt := st.chat_input("What is up?"):
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         assistant_response = random.choice(
#             [
#                 "Hello there! How can I assist you today?",
#                 "Hi, human! Is there anything I can help you with?",
#                 "Do you need help?",
#             ]
#         )
#         # Simulate stream of response with milliseconds delay
#         for chunk in assistant_response.split():
#             full_response += chunk + " "
#             time.sleep(0.05)
#             # Add a blinking cursor to simulate typing
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})



import openai
import streamlit as st

st.title("Personal ChatGPT")
st.sidebar.title("Options")
model_name = st.sidebar.radio("Choose model:",
                                ("gpt-3.5-turbo-1106", "gpt-4", "gpt-4-1106-preview"), index=2)
# temperature = st.sidebar.slider("Temperature:", min_value=0.,
#                                 max_value=2., value=0., step=0.2)
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

if prompt := st.chat_input("What is up?"):
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