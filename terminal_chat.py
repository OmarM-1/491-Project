import requests
import uuid

URL = "http://127.0.0.1:8001/chat"
SESSION_ID = str(uuid.uuid4())

def run():
    print("Calorie Chat (type 'exit' to quit)\n")
    while True:
        msg = input("You: ").strip()
        if msg.lower() in {"exit", "quit"}:
            break

        r = requests.post(URL, json={"message": msg, "session_id": SESSION_ID})
        r.raise_for_status()
        data = r.json()

        print(f"\nAssistant ({data['route']}):")
        print(data["answer"])
        print("-" * 60)

if __name__ == "__main__":
    run()

