import os
import discord

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
# import psycopg2
import datetime
import threading
import concurrent.futures
import re 

from functions.query import query_data
from functions.summary import summarize_conversation
from functions.search import search_conversation, search_conversation_quick

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

# Memory: {user_id: [(role, message, timestamp)]}
user_chat_history = {}
total_chat_history = {}
total_chat_embeddings = {}
cached_chat_history = {}

# dictionary keyed by user_id, where each value is a list of query strings made by that user in the current session.
user_query_session_history = {}  # { user_id: [query_1, query_2, ...] }

# used to determine if a user’s message that does not start with "query" should be interpreted as a follow-up to a previous query.
last_command_type = {}  # { user_id: "query" or None }

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
    # Send a message to all text channels the bot can access
    for guild in client.guilds:
        for channel in guild.text_channels:
            try:
                await channel.send("🤖 I am up and ready!\n For a full list of commands type: `help`")
                break  # Only send to the first accessible text channel per guild
            except Exception:
                continue

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_id      = message.author.id    #unique Discord user ID of the message sender.
    user_name    = message.author.mention   #string to mention/tag the user in a message.
    user_message = message.content.strip()  #actual text content of the message the user sent.
    channel_id   = message.channel.id   #The unique ID of the channel where the message was sent.
    now = datetime.datetime.utcnow()

    # Initialize histories
    user_chat_history.setdefault(user_id, [])
    total_chat_history.setdefault(channel_id, [])
    total_chat_embeddings.setdefault(channel_id, InMemoryVectorStore(embedding=embedding_model))
    cached_chat_history.setdefault(channel_id, [])

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
        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        cached_chat_history[channel_id].append((user_name, user_message, now))
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
        total_chat_history[channel_id].append(("Bot", summary, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=summary, metadata={"user": "Bot", "timestamp": now})])
        cached_chat_history[channel_id].append(("Bot", summary, now))
        return

    # ---- Help command --------------------------
    if user_message.lower() == "help":
        await message.channel.send(
            "# General Help:\n"
            "Use `ask: <your question>` to ask the bot a question.\n"
            "Use `summary` to get a summary of the conversation.\n"
            "Use `summary: last X [minute|hour|day|week|month|year]` for a time-limited summary.\n"
            "Use `search: <terms>` to search for terms in the conversation.\n"
            "Use `query: <your query>` to run a SQL-like query on the conversation.\n"
            "After typing `query:`, you can ask follow-up questions directly without typing `query:` again.\n"
            "Use `clear` to clear your memory.\n"
            "Use `exit` to stop the bot.\n"
            "# Testing Help:\n"
            "Use `gen_chat` to generate a new conversation with timestamps (timestamps are from 6/19/2025).\n"
            #"Use `gen_no_time` to generate a new conversation without timestamps.\n"
            "Use `test` to test the bot's response.\n"
            "Use `show_history` to display the current conversation history.\n"
            "Use `where_am_i` to find out which channel you are in.\n"
            "Use `help` to see this message again.\n"
        )
        return
    
    # ---- Summary command ------------------------
    if user_message.lower() == "summary":
        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        cached_chat_history[channel_id].append((user_name, user_message, now))
        # summary = summarize_conversation(user_chat_history[user_id])
        summary = summarize_conversation(total_chat_history[channel_id])
        response = split_response(summary)
        await message.channel.send(f"📋 Summary:\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        total_chat_history[channel_id].append(("Bot", summary, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=summary, metadata={"user": "Bot", "timestamp": now})])
        cached_chat_history[channel_id].append(("Bot", summary, now))
        return

    # ---- Query handler ---------------------
    if user_message.lower().startswith("query: "):

        user_query = user_message[7:].strip()

        # Initialize session history
        user_query_session_history.setdefault(user_id, [])

        # Store current query in session history
        user_query_session_history[user_id].append(user_query)

        # Combine all queries from this session as context
        session_history = user_query_session_history[user_id]

        # Pass session history (list of past queries) to query_data
        texts = query_data(user_id, user_query, session_history=session_history)

        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)
        # Save current message to history as usual
        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        cached_chat_history[channel_id].append((user_name, user_message, now))

        # Save bot response to history
        total_chat_history[channel_id].append(("Bot", texts, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=str(texts), metadata={"user": "Bot", "timestamp": now})])
        cached_chat_history[channel_id].append(("Bot", str(texts), now))

        last_command_type[user_id] = "query"
        return

    # ---- Follow-up Query Handler (no "query:" prefix) ---------------------
    if last_command_type.get(user_id) == "query" and not user_message.lower().startswith(("ask:", "summary", "search:", "help", "exit", "clear", "show_history", "test", "show_embeds", "gen_chat", "where_am_i")):
        user_query = user_message.strip()
        user_query_session_history.setdefault(user_id, [])
        user_query_session_history[user_id].append(user_query)
        session_history = user_query_session_history[user_id]
        texts = query_data(user_id, user_query, session_history=session_history)

        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)

        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        cached_chat_history[channel_id].append((user_name, user_message, now))

        total_chat_history[channel_id].append(("Bot", texts, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=str(texts), metadata={"user": "Bot", "timestamp": now})])
        cached_chat_history[channel_id].append(("Bot", str(texts), now))
        return

    # ---- Existing /search handler ---------------
    if user_message.lower().startswith("search: "):
        terms = user_message[len("search: "):].strip()

        await message.channel.send(f"🔎 Searching for: `{terms}`")
        quick_result = search_conversation_quick(total_chat_embeddings[channel_id], terms)
        await message.channel.send(f"🔎 Search result:\n{quick_result}")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(search_conversation, total_chat_embeddings[channel_id], terms, cached_chat_history[channel_id], channel_id, quick_result)
            
            try:
                total_result = future.result(timeout=30)
                if total_result:
                    await message.channel.send(total_result)
            except concurrent.futures.TimeoutError:
                print("Long search operation timed out, using only quick response instead.")
            except Exception as e:
                await message.channel.send(f"❌ Error: {e}")

        cached_chat_history[channel_id] = []
        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        cached_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_history[channel_id].append(("Bot", quick_result, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=quick_result, metadata={"user": "Bot", "timestamp": now})])
        cached_chat_history[channel_id].append(("Bot", quick_result, now))
        if total_result:
            total_chat_history[channel_id].append(("Bot", total_result, now))
            total_chat_embeddings[channel_id].add_documents([Document(page_content=total_result, metadata={"user": "Bot", "timestamp": now})])
            cached_chat_history[channel_id].append(("Bot", total_result, now))
        return

    # ---- Testing utilities ----------------------
    if user_message.lower() == "test":
        await message.channel.send(f"Your ID: {user_id}")
        await message.channel.send(f"{user_name} just sent me a message")
        return

    # Show embeds might need work on because i'm not sure how to display the embeddings in a readable format - kyle
    if user_message.lower() == "show_embeds":
        print(total_chat_embeddings[channel_id])
        response = split_response(str(total_chat_embeddings[channel_id]), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    if user_message.lower() == "show_history":
        print(total_chat_history[channel_id])
        print(user_query_session_history)
        response = split_response(str(total_chat_history[channel_id]), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    if user_message.lower() == "where_am_i":
        await message.channel.send(f"Channel: {message.channel.name} ({channel_id})")
        return
    
    if user_message.lower() == "gen_chat":
        total_chat_history[channel_id] = []
        total_chat_history[channel_id] = [('<@458754358004416524>', 'how is the project coming along?', datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('Bot', "It's going quite well, actually! We've made excellent progress on the new data pipeline integration, even hitting a few of our stretch goals. How's the client-facing dashboard development coming along for you?", datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('<@458754358004416524>', 'the dashboard is going great as of now. we just finished up some of the nicer UI elements on it as well. The clients are loving it so far.', datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('Bot', "That's fantastic news about the UI elements! It's always great when the clients are actively engaged and happy. Are you still on track to integrate the real-time analytics module by end of next week, or has the UI work shifted that timeline at all?", datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('<@458754358004416524>', 'currently the real-time analytics are going through some problems with the database. we need to get some more permissions from Mr. Bob about that though. How is your team doing with the presentation for tomorrow?', datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('Bot', "Oh, the presentation for tomorrow is looking solid! We just wrapped up the final review of the Q3 projections and the new workflow diagrams. Everything's synced up. Did you manage to get those updated user testimonials from marketing for the demo section?", datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('<@458754358004416524>', 'the demo is awesome right now. its all interactive and has been tested thoroughly.', datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('Bot', "That's incredible to hear about the demo! Interactive and thoroughly tested is exactly what we need. So, are we still planning to showcase the new dynamic filtering capabilities within that interactive demo, or have you focused more on the core user journeys for this iteration?", datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('<@458754358004416524>', 'yes we have been talking back and forth with the client. They are changing some stuff up on us though. because of that make sure that you scrap that last slide over the current backend page for the videos. We are changing that entirely but have no clue to what as of now.', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('Bot', 'Understood on scrapping that slide for the backend video page – definitely noted. Thanks for the heads-up on the client changes, that sounds like it could be a significant pivot. Is this likely to affect the proposed integration of the secure asset management module, or is that still on track?', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('<@458754358004416524>', "everything seems to still be on-track. But i'm not sure how we want to go about figuring this all out before the demo. Do you have anyone you can ask about it to try getting an answer faster?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('Bot', "That's a good point about the demo timeline. For figuring out the new direction quickly, I'd suggest reaching out to Alex in Product Strategy. He's usually got a great bird's-eye view of client needs and can often provide a solid temporary placeholder or a quick path forward. Have you already looped in the QA team about the potential impact of these changes on their final testing pass?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('<@458754358004416524>', "Thats true! Alex and Steve can for sure help us out with this. They can reach out and get us more of an idea on what we are supposed to do next regarding the demo. As for testing I think that we should try getting someone on John's testing team to figure that out.", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('Bot', "Good call on Alex and Steve – they'll definitely help clarify the demo direction. And getting someone from John's team on the testing impact is smart. Have you thought about how we'll communicate these potential last-minute changes to the training materials team, especially if the demo flow shifts significantly?", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('<@458754358004416524>', "I've just sent them a message about this topic. Steve says that they are down to help us out!", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('Bot', "That's excellent news about Steve and the training materials team! Glad they're on board to help. Does this mean we'll need to re-evaluate the integration points for the new user onboarding flow, given the potential demo changes?", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('<@458754358004416524>', 'honestly i think everything is looking good for now. ill call you later. Goodbye!', datetime.datetime(2025, 6, 19, 19, 51, 17, 303753)), ('Bot', "Sounds good! Glad to hear everything's shaping up. Talk later then! Just to confirm for my planning, are we still on track to get the final sign-off on the API documentation updates by end of day today?", datetime.datetime(2025, 6, 19, 19, 51, 17, 303753))]
        await message.channel.send("✅🕒 Memory cleared and generated with time!")
        return

    # if user_message.lower() == "gen_no_time":
    #     total_chat_history[channel_id] = []
    #     total_chat_history[channel_id] = [('<@458754358004416524>', 'hi mr bot'), ('Bot', 'Hello! How can I help you today?'), ('<@458754358004416524>', 'i really like the color blue'), ('Bot', "That's wonderful! Blue is a beautiful color, often associated with peace and calm. Do you have a favorite shade?"), ('<@458754358004416524>', 'do you like the color blue too?'), ('Bot', 'As an AI, I don\'t have personal preferences or feelings, so I don\'t "like" colors in the way humans do. However, I can understand why you find blue appealing!'), ('<@458754358004416524>', 'i think my fav part about blue is that it is like the sky'), ('Bot', "That's a lovely connection! The sky's various shades of blue are truly captivating."), ('<@458754358004416524>', 'the dragon is also like the sky'), ('<@458754358004416524>', 'that is because they are able to fly'), ('Bot', "That's an interesting connection! Many dragons are often depicted as creatures of the sky, soaring high above."), ('Bot', 'I understand your connection! Dragons soaring through the air do seem like a part of the sky.'), ('<@458754358004416524>', 'dragons that fly are so cool'), ('Bot', 'They certainly are! The image of them soaring through the air is quite impressive.'), ('<@458754358004416524>', 'dragons remind me of that one movie'), ('Bot', 'Oh, which movie are you thinking of?'), ('<@458754358004416524>', 'the one where they train dragons'), ('Bot', 'You\'re likely thinking of "How to Train Your Dragon"! It\'s a very popular film series.'), ('<@458754358004416524>', 'yea toothless is my fav dragon of all time'), ('Bot', "Toothless is a fantastic choice! He's definitely one of the most beloved dragons from that series."), ('<@458754358004416524>', 'what should we talk about?'), ('Bot', 'We could continue talking about mythical creatures, or perhaps shift to other favorite movies or stories you enjoy.'), ('<@458754358004416524>', 'i like mythical creatures, whats ur fav animal'), ('Bot', "That's a fun topic! As an AI, I don't have a favorite animal in the way humans do. However, I'm curious, what's your favorite mythical creature?"), ('<@458754358004416524>', 'i think unicorns are funny looking'), ('Bot', 'They do have a unique appearance, with their single horn.'), ('<@458754358004416524>', 'they are just horses with a horn'), ('Bot', "That's a pretty accurate way to describe them! Their defining feature is indeed that distinctive horn."), ('<@458754358004416524>', 'i like league of legends'), ('Bot', "Ah, League of Legends! That's a very popular game. What do you enjoy most about it?"), ('<@458754358004416524>', 'that is a fun game to me'), ('Bot', "That's great to hear! What aspects of the game do you find most enjoyable?"), ('<@458754358004416524>', 'i play aphelios, and that is why i like it so much'), ('Bot', 'Aphelios is a really unique champion in League of Legends! It makes perfect sense that playing a character you enjoy so much would make the game more fun for you. What do you like most about playing him?'), ('<@458754358004416524>', 'aphelios is just a fun champion to play'), ('Bot', "That's great! What about his playstyle makes him so enjoyable for you?")]
    #     await message.channel.send("❌🕒 Memory cleared and generated without time!")
    #     return

    

    # Only allow default chat if message starts with 'ask: '
    if user_message.lower().startswith("ask: "):
        user_real_message = user_message[5:].strip()
        total_chat_history[channel_id].append((user_name, user_real_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_real_message, metadata={"user": user_name, "timestamp": now})])
        user_chat_history[user_id].append((user_name, user_real_message, now))
        cached_chat_history[channel_id].append((user_name, user_real_message, now))

        # Build prompt

        
        full_prompt = f"""
        Be Formal with your replies. This is a work environment.
        I do NOT want any bullet points when you respond to me unless I ask you for them.
        These replies MUST be short as to not clutter the text chat.
        All replies must be readable. Now respond to this given the previous instructions:\n
        """

        # Test Data Creation Prompt:
        # full_prompt = "You are my coworker and we are having a conversation about one of our projects. make up details about it when i ask you something. never say you dont know something, always answer. also come up with fake follow up questions for me if possible. This conversation will be used as training data, so do not worry about accuracy. Make sure to keep your responses limited in length as to not clutter the text chat. Now using these instructions reply to me:\n"

        for role, msg, _ in total_chat_history[channel_id]:
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
        total_chat_history[channel_id].append(("Bot", bot_reply, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=bot_reply, metadata={"user": "Bot", "timestamp": now})])
        user_chat_history[user_id].append(("Bot", bot_reply, now))
        cached_chat_history[channel_id].append(("Bot", bot_reply, now))
        return
    else:
        total_chat_history[channel_id].append((user_name, user_message, now))
        total_chat_embeddings[channel_id].add_documents([Document(page_content=user_message, metadata={"user": user_name, "timestamp": now})])
        user_chat_history[user_id].append((user_name, user_message, now))
        cached_chat_history[channel_id].append((user_name, user_message, now))
client.run(discord_token)
