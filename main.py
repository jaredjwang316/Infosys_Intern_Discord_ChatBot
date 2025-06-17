import os
from google import genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ GOOGLE_API_KEY not found in .env")
    exit()

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# Manual chat history
chat_history = []

print("🤖 Gemini Chat Bot is running. Type 'summary' to summarize or 'exit' to quit.\n")

def summarize_conversation(history):
    prompt = "Summarize the following conversation:\n"
    for role, message in history:
        prompt += f"{role}: {message}\n"
    response = model.generate_content(prompt)
    return response.text.strip()

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "exit":
        print("👋 Goodbye!")
        break
    elif user_input.strip().lower() == "summary":
        print("📋 Summary:")
        print(summarize_conversation(chat_history))
        continue

    # Add user input to chat history
    chat_history.append(("User", user_input))

    # Create a prompt with entire conversation
    full_prompt = ""
    for role, msg in chat_history:
        full_prompt += f"{role}: {msg}\n"
    full_prompt += "Bot:"

    # Generate response
    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text.strip()
    except Exception as e:
        print("❌ Error:", e)
        continue

    print("Bot:", bot_reply)
    chat_history.append(("Bot", bot_reply))