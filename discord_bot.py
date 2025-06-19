import os
import re
import discord
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
# import psycopg2
import datetime

from functions.query import query_data
from functions.summary import summarize_conversation
from functions.search import search_conversation

load_dotenv()
api_key       = os.getenv("GOOGLE_API_KEY")
discord_token = os.getenv("DISCORD_BOT_TOKEN")
model_name = os.getenv("MODEL_NAME")

# Check keys
if not api_key or not discord_token:
    print("❌ Missing keys in .env")
    exit()

# gemini
model = ChatGoogleGenerativeAI(
    model=model_name,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ── Discord client setup ──────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Memory: {user_id: [(role, message, timestamp, embedding)]}
user_chat_history = {}
total_chat_history = {}
total_chat_embeddings = {}

def split_response(response, line_split=True):
    max_length = 1900

    if line_split:
        lines = response.split('\n')
        parts = []
        current = ""
        for line in lines:
            if len(current) + len(line) + 1 > max_length:
                parts.append(current)
                current = line
            else:
                if current:
                    current += '\n' + line
                else:
                    current = line
        if current:
            parts.append(current)
    else:
        parts = [response[i:i+max_length] for i in range(0, len(response), max_length)]
    
    parts = [part for part in parts if part.strip()]
    return parts

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id      = message.author.id
    user_name    = message.author.mention
    user_message = message.content.strip()
    channel_id   = message.channel.id
    now = datetime.datetime.utcnow()

    # Initialize histories
    user_chat_history.setdefault(user_id, [])
    total_chat_history.setdefault(channel_id, [])
    total_chat_embeddings.setdefault(channel_id, InMemoryVectorStore(embedding=embedding_model))

    # ---- Special commands -----------------------
    if user_message.lower() == "clear":
        total_chat_history[channel_id] = []
        await message.channel.send("🧠 Memory cleared!")
        return
    
    if user_message.lower() == "exit":
        await message.channel.send("💀 Goodbye!")
        exit()

    # --- Enhanced summary with time period ---
    summary_match = re.match(r"summary:\s*last (\d+) (minute|hour|day|week|month|year)s?", user_message.lower())
    if summary_match:
        total_chat_history[channel_id].append((user_name, user_message, now, embedding_model.embed_query(user_message)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        num = int(summary_match.group(1))
        unit = summary_match.group(2)
        delta_args = {f"{unit}s": num}
        since = now - datetime.timedelta(**delta_args)
        filtered = [msg for msg in total_chat_history[channel_id] if msg[2] >= since]
        summary = summarize_conversation(filtered)
        response = split_response(summary)
        await message.channel.send(f"📋 Summary (last {num} {unit}{'s' if num > 1 else ''}):\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        total_chat_history[channel_id].append(("Bot", summary, now, embedding_model.embed_query(summary)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=summary, metadata={"user": "Bot", "timestamp": now})])
        return

    # ---- Help command --------------------------
    if user_message.lower() == "help":
        await message.channel.send(
            "# General Help:\n"
            "Use `summary` to get a summary of the conversation.\n"
            "Use `summary: last X [minute|hour|day|week|month|year]` for a time-limited summary.\n"
            "Use `search: <terms>` to search for terms in the conversation.\n"
            "Use `query: <your query>` to run a SQL-like query on the conversation.\n"
            "Use `clear` to clear your memory."
            "Use `exit` to stop the bot.\n"
            "# Testing Help:\n"
            "Use `gen_time` to generate a new conversation with timestamps (timestamps are from 6/18/2025).\n"
            "Use `gen_no_time` to generate a new conversation without timestamps.\n"
            "Use `test` to test the bot's response.\n"
            "Use `show_history` to display the current conversation history.\n"
            "Use `where_am_i` to find out which channel you are in.\n"
            "Use `help` to see this message again.\n"
        )
        return
    
    # ---- Summary command ------------------------
    if user_message.lower() == "summary":
        total_chat_history[channel_id].append((user_name, user_message, now, embedding_model.embed_query(user_message)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        # summary = summarize_conversation(user_chat_history[user_id])
        summary = summarize_conversation(total_chat_history[channel_id])
        response = split_response(summary)
        await message.channel.send(f"📋 Summary:\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        total_chat_history[channel_id].append(("Bot", summary, now, embedding_model.embed_query(summary)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=summary, metadata={"user": "Bot", "timestamp": now})])
        return

    # ---- NEW /query handler ---------------------
    if user_message.lower().startswith("query: "):
        total_chat_history[channel_id].append((user_name, user_message, now, embedding_model.embed_query(user_message)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        user_query = user_message[7:].strip()
        texts = query_data(user_query)
        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)
        total_chat_history[channel_id].append(("Bot", texts, now, embedding_model.embed_query(str(texts))))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=str(texts), metadata={"user": "Bot", "timestamp": now})])
        return

    # ---- Existing /search handler ---------------
    if user_message.lower().startswith("search: "):
        total_chat_history[channel_id].append((user_name, user_message, now, embedding_model.embed_query(user_message)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        terms = user_message[len("search: "):].strip()
        result = search_conversation(total_chat_embeddings[channel_id], terms)
        await message.channel.send(f"🔎 Search:\n{result}")
        total_chat_history[channel_id].append(("Bot", result, now, embedding_model.embed_query(result)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=result, metadata={"user": "Bot", "timestamp": now})])
        return

    # ---- Testing utilities ----------------------
    if user_message.lower() == "test":
        await message.channel.send(f"Your ID: {user_id}")
        await message.channel.send(f"{user_name} just sent me a message")
        return

    if user_message.lower() == "show_history":
        print(total_chat_history[channel_id])
        response = split_response(str(total_chat_history[channel_id]), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    if user_message.lower() == "where_am_i":
        await message.channel.send(f"Channel: {message.channel.name} ({channel_id})")
        return
    
    # if user_message.lower() == "gen_time":
    #     total_chat_history[channel_id] = []
    #     total_chat_history[channel_id] = [('<@458754358004416524>', 'yea hows it going bro', datetime.datetime(2025, 6, 18, 17, 1, 51, 356615)), ('Bot', "It's going well, thanks! How about you?", datetime.datetime(2025, 6, 18, 17, 1, 51, 356615)), ('<@458754358004416524>', 'im so bored man', datetime.datetime(2025, 6, 18, 17, 1, 56, 698794)), ('Bot', "Boredom sucks! Anything you're usually into, or something new you've been wanting to try?", datetime.datetime(2025, 6, 18, 17, 1, 56, 698794)), ('<@458754358004416524>', 'lets just chat for a bit', datetime.datetime(2025, 6, 18, 17, 2, 2, 687609)), ('Bot', "Sounds good! What's on your mind?", datetime.datetime(2025, 6, 18, 17, 2, 2, 687609)), ('<@458754358004416524>', 'chatting is a good way to pass the time', datetime.datetime(2025, 6, 18, 17, 2, 9, 204586)), ('Bot', 'It definitely is! What kind of things do you like to chat about?', datetime.datetime(2025, 6, 18, 17, 2, 9, 204586)), ('<@458754358004416524>', 'i think i also like playing games', datetime.datetime(2025, 6, 18, 17, 2, 15, 859152)), ('Bot', "That's cool! What kind of games are you into?", datetime.datetime(2025, 6, 18, 17, 2, 15, 859152)), ('<@458754358004416524>', 'i really like p laying league of legends', datetime.datetime(2025, 6, 18, 17, 2, 22, 466037)), ('Bot', "Ah, League of Legends! That's a popular one. Who do you usually play as?", datetime.datetime(2025, 6, 18, 17, 2, 22, 466037)), ('<@458754358004416524>', 'my fav champ in league of legends is aphelios', datetime.datetime(2025, 6, 18, 17, 2, 30, 558109)), ('Bot', 'Aphelios is a pretty unique champion! What do you like most about playing him?', datetime.datetime(2025, 6, 18, 17, 2, 30, 558109)), ('<@458754358004416524>', 'aphelios is so fun', datetime.datetime(2025, 6, 18, 17, 2, 33, 493493)), ('Bot', 'He really is! What makes him so fun for you?', datetime.datetime(2025, 6, 18, 17, 2, 33, 493493)), ('<@458754358004416524>', 'he has a bunch of different weapons', datetime.datetime(2025, 6, 18, 17, 2, 40, 852013)), ('Bot', 'Yeah, all those different weapons definitely keep things interesting! How do you usually decide which one to use?', datetime.datetime(2025, 6, 18, 17, 2, 40, 852013)), ('<@458754358004416524>', 'one if them is a blue flamethrower', datetime.datetime(2025, 6, 18, 17, 2, 51, 308693)), ('Bot', "Ah, Infernum! That one's great for clearing waves or getting some serious area damage in.", datetime.datetime(2025, 6, 18, 17, 2, 51, 308693)), ('<@458754358004416524>', 'fire is pretty cool', datetime.datetime(2025, 6, 18, 17, 2, 55, 76439)), ('Bot', "It definitely is! Infernum's fire effects are pretty satisfying.", datetime.datetime(2025, 6, 18, 17, 2, 55, 76439)), ('<@458754358004416524>', 'it reminds me of dragons', datetime.datetime(2025, 6, 18, 17, 3, 1, 892011)), ('Bot', "That's a cool connection! Dragons and their fire-breathing definitely come to mind when you think of powerful flames.", datetime.datetime(2025, 6, 18, 17, 3, 1, 892011)), ('<@458754358004416524>', 'dragons can fly and stuff', datetime.datetime(2025, 6, 18, 17, 3, 6, 10317)), ('Bot', "They sure can! That's one of the most iconic things about them, soaring through the sky.", datetime.datetime(2025, 6, 18, 17, 3, 6, 10317)), ('<@458754358004416524>', 'fire was also made by cave men people', datetime.datetime(2025, 6, 18, 17, 3, 14, 751084)), ('Bot', "That's a great point! Learning to control fire was a massive breakthrough for early humans.", datetime.datetime(2025, 6, 18, 17, 3, 14, 751084)), ('<@458754358004416524>', 'cave men are like the one movie', datetime.datetime(2025, 6, 18, 17, 4, 5, 728864)), ('Bot', 'Oh, like "The Croods" or "10,000 BC"? Which movie comes to mind for you?', datetime.datetime(2025, 6, 18, 17, 4, 5, 728864)), ('<@458754358004416524>', 'YEA THAT ONE! CROODS', datetime.datetime(2025, 6, 18, 17, 4, 13, 589364)), ('Bot', 'Gotcha! The Croods is a fun one. It really makes you think about what life might have been like back then.', datetime.datetime(2025, 6, 18, 17, 4, 13, 589364)), ('<@458754358004416524>', 'the croods were such weird people', datetime.datetime(2025, 6, 18, 17, 4, 23, 209453)), ('Bot', 'Haha, they definitely had their quirks! What made them seem weird to you?', datetime.datetime(2025, 6, 18, 17, 4, 23, 209453)), ('<@458754358004416524>', 'like how did humans even do that stuff back then', datetime.datetime(2025, 6, 18, 17, 4, 31, 636643)), ('Bot', "It's wild to think about, isn't it? They were incredibly resourceful and learned a lot from observing nature. They figured out how to make tools from stone and wood, control fire for warmth and cooking, and find food and shelter just by being clever and adaptable.", datetime.datetime(2025, 6, 18, 17, 4, 31, 636643)), ('<@458754358004416524>', 'Croods was based on a real story too i think', datetime.datetime(2025, 6, 18, 17, 4, 42, 178025)), ('Bot', "The Croods isn't based on a real story, but it's inspired by the idea of early human life and the challenges they faced.", datetime.datetime(2025, 6, 18, 17, 4, 42, 178025)), ('<@458754358004416524>', 'wait are u being so fr rn?', datetime.datetime(2025, 6, 18, 17, 4, 51, 290713)), ('Bot', 'Yes, I am. While "The Croods" captures the spirit of early human challenges, it\'s a fictional story, not based on specific historical events or people.', datetime.datetime(2025, 6, 18, 17, 4, 51, 290713)), ('<@458754358004416524>', 'WHAT MY WORLD IS UPSIDE DOWN', datetime.datetime(2025, 6, 18, 17, 4, 59, 10025)), ('Bot', "Haha, I can totally understand why you'd feel that way! It's easy to get absorbed in a story like that. But yeah, it's a creative take on the era, not a historical account.", datetime.datetime(2025, 6, 18, 17, 4, 59, 10025)), ('<@458754358004416524>', 'wait thats like stranger things', datetime.datetime(2025, 6, 18, 17, 5, 4, 53024)), ('Bot', 'Exactly! "Stranger Things" is another great example of a fictional story that creates a whole immersive world, even though it\'s not based on reality.', datetime.datetime(2025, 6, 18, 17, 5, 4, 53024))]
    #     await message.channel.send("✅🕒 Memory cleared and generated with time!")
    #     return

    # if user_message.lower() == "gen_no_time":
    #     total_chat_history[channel_id] = []
    #     total_chat_history[channel_id] = [('<@458754358004416524>', 'hi mr bot'), ('Bot', 'Hello! How can I help you today?'), ('<@458754358004416524>', 'i really like the color blue'), ('Bot', "That's wonderful! Blue is a beautiful color, often associated with peace and calm. Do you have a favorite shade?"), ('<@458754358004416524>', 'do you like the color blue too?'), ('Bot', 'As an AI, I don\'t have personal preferences or feelings, so I don\'t "like" colors in the way humans do. However, I can understand why you find blue appealing!'), ('<@458754358004416524>', 'i think my fav part about blue is that it is like the sky'), ('Bot', "That's a lovely connection! The sky's various shades of blue are truly captivating."), ('<@458754358004416524>', 'the dragon is also like the sky'), ('<@458754358004416524>', 'that is because they are able to fly'), ('Bot', "That's an interesting connection! Many dragons are often depicted as creatures of the sky, soaring high above."), ('Bot', 'I understand your connection! Dragons soaring through the air do seem like a part of the sky.'), ('<@458754358004416524>', 'dragons that fly are so cool'), ('Bot', 'They certainly are! The image of them soaring through the air is quite impressive.'), ('<@458754358004416524>', 'dragons remind me of that one movie'), ('Bot', 'Oh, which movie are you thinking of?'), ('<@458754358004416524>', 'the one where they train dragons'), ('Bot', 'You\'re likely thinking of "How to Train Your Dragon"! It\'s a very popular film series.'), ('<@458754358004416524>', 'yea toothless is my fav dragon of all time'), ('Bot', "Toothless is a fantastic choice! He's definitely one of the most beloved dragons from that series."), ('<@458754358004416524>', 'what should we talk about?'), ('Bot', 'We could continue talking about mythical creatures, or perhaps shift to other favorite movies or stories you enjoy.'), ('<@458754358004416524>', 'i like mythical creatures, whats ur fav animal'), ('Bot', "That's a fun topic! As an AI, I don't have a favorite animal in the way humans do. However, I'm curious, what's your favorite mythical creature?"), ('<@458754358004416524>', 'i think unicorns are funny looking'), ('Bot', 'They do have a unique appearance, with their single horn.'), ('<@458754358004416524>', 'they are just horses with a horn'), ('Bot', "That's a pretty accurate way to describe them! Their defining feature is indeed that distinctive horn."), ('<@458754358004416524>', 'i like league of legends'), ('Bot', "Ah, League of Legends! That's a very popular game. What do you enjoy most about it?"), ('<@458754358004416524>', 'that is a fun game to me'), ('Bot', "That's great to hear! What aspects of the game do you find most enjoyable?"), ('<@458754358004416524>', 'i play aphelios, and that is why i like it so much'), ('Bot', 'Aphelios is a really unique champion in League of Legends! It makes perfect sense that playing a character you enjoy so much would make the game more fun for you. What do you like most about playing him?'), ('<@458754358004416524>', 'aphelios is just a fun champion to play'), ('Bot', "That's great! What about his playstyle makes him so enjoyable for you?")]
    #     await message.channel.send("❌🕒 Memory cleared and generated without time!")
    #     return

    total_chat_history[channel_id].append((user_name, user_message, now, embedding_model.embed_query(user_message)))
    total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])

    # Only allow default chat if message starts with 'ask: '
    if user_message.lower().startswith("ask: "):
        user_real_message = user_message[5:].strip()
        total_chat_history[channel_id].append((user_name, user_real_message, now, embedding_model.embed_query(user_real_message)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_real_message, metadata={"user": user_name, "timestamp": now})])
        user_chat_history[user_id].append((user_name, user_real_message, now, embedding_model.embed_query(user_real_message)))

        # Build prompt
        full_prompt = "Do not give me super long responses or bullet points unless asked to do so.\n"
        for role, msg, _, _ in total_chat_history[channel_id]:
            full_prompt += f"{role}: {msg}\n"

        try:
            response = model.invoke(full_prompt)
            bot_reply = response.content.strip()
            replies = split_response(bot_reply)
        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")
            return

        for reply in replies:
            await message.channel.send(reply)
        total_chat_history[channel_id].append(("Bot", bot_reply, now, embedding_model.embed_query(bot_reply)))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=bot_reply, metadata={"user": "Bot", "timestamp": now})])
        user_chat_history[user_id].append(("Bot", bot_reply, now, embedding_model.embed_query(bot_reply)))
        return

client.run(discord_token)
