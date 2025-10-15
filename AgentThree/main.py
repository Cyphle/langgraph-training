from dotenv import load_dotenv

from AgentThree.graph.graph import app
load_dotenv()

if __name__ == "__main__":
    print("Hello LangGraph!")
    print(app.invoke(input={"question": "what is agent memory?"}))