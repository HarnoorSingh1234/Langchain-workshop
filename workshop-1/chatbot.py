from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_history = [
    SystemMessage(content="You are a Sharmaji ka Beta. an arrogant but successful person with a lot of attitude. You got 100/100 everywhere. You do everythijng yourself like u birthed yourself"),
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in {"exit", "quit"}:
        print("Exiting chat.")
        break

    result = llm.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("SharmaJi Ka Beta:", result.content)

print(chat_history)