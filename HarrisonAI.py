import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import streamlit as st
from streamlit_chat import message as st_message
import requests

EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"
url = "http://172.20.2.40:5000/api/v1/related-strings"


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Harrison AI")
# expander = st.expander(label='set your open ai API key')
# with expander:
#     openai.api_key = st.text_input("Enter your OpenAI API key:")
openai.api_key = st.secrets["APIKEY"]


def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    query_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    # strings, relatednesses = strings_ranked_by_relatedness(query_embedding_response)
    params = {"top_n": 10}
    response = requests.post(url, json={"query": query_embedding_response, "top_n": params["top_n"]})
    if response.status_code == 200:
        # Extract the strings and relatednesses from the response JSON
        result = response.json()
        strings = result["strings"]
        relatednesses = result["relatednesses"]

    else:
        print("Error:", response.content)

    introduction = 'Use the below articles from Harrison’s Principles of Internal Medicine 21st Edition to answer the subsequent question at the end write the page number you used and table name,answer in quesion language. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\n   : \n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    
    print( "Used Token: ",num_tokens(message + question))
    return message + question

def ask(
    query: str,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions based on Harrison’s Principles of Internal Medicine 21st Edition. You should mention the page numbers you used."},
        {"role": "user", "content": message},
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
        # stream=True
    )
    
    response_message = response["choices"][0]["message"]["content"]
    
    # print (response_message) 
    return response_message

def submitAPIKEY(key):
    print (key)
    openai.api_key = key


for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i)) #unpacking

def generate_answer():
    user_message = st.session_state.input_text
    st.session_state.history.append({"message": user_message, "is_user": True})
    # prompt = "\n".join([m['message'] for m in st.session_state.history])
    # with st.spinner("Thinking..."):
    result = ask(user_message)
    # try:
    # except:
    #     st.error('An error occurred, check your API key')
    message_bot = result
    st.session_state.history.append({"message": message_bot, "is_user": False})

input_value = st.text_input("Type your message:", key="input_text")
st.button('Send', on_click=generate_answer)


# while True:
#   user_input = input("Ask: ")
#   if user_input == "quit":
#     break
#   try: 
#     ask(user_input)
#   except:
#      print("Error Occurred")
