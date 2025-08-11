# Quick CLI testing without launching Streamlit

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

# Load environment variables from a .env file if present
load_dotenv()

# Validate that the Google API key is available
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. In Command Prompt, run: set GOOGLE_API_KEY=your_key or use setx for persistence."
    )

# define the prompt template
system_template = "You are a knowlegeable history teacher. You are given a question and you need to answer it in a way that is easy to understand and engaging for a 10 year old."

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# initialize the gemini model (reads GOOGLE_API_KEY from environment)
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
)

# create the chain
chain = prompt_template | model

# Add in-memory chat history and wrap the chain to be context-aware
session_histories = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# interactive chat loop 
print("Welcome to the history chatbot! Type 'exit' to end the conversation.")
while True:
    user_input = input("Your question: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    try:
        response = conversation.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": "default"}},
        )
        print("AI: ", response.content)
    except Exception as e:
        print(f"Error: {e}, plz check your api key and try again")



