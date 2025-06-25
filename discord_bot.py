import os
import discord

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import datetime
import concurrent.futures
import re

from functions.query import query_data
from functions.summary import summarize_conversation, summarize_conversation_by_time
from functions.search import search_conversation, search_conversation_quick
from local_memory import LocalMemory

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
local_memory = LocalMemory()

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

    # --- Enhanced summary -------------------------------------------------------------------------------------------------------------------
    summary_match = re.match(r"summary:\s*last (\d+) (minute|hour|day|week|month|year)s?", user_message.lower())
    if summary_match:
        local_memory.add_message(channel_id, user_id, user_message)

        local_memory.store_all_in_long_term_memory()
        num = int(summary_match.group(1))
        unit = summary_match.group(2)
        delta_args = {f"{unit}s": num}
        since = now - datetime.timedelta(**delta_args)
        summary = summarize_conversation_by_time(
            channel_id,
            start_time=since,
            end_time=now
        )
        response = split_response(summary)
        await message.channel.send(f"📋 Summary (last {num} {unit}{'s' if num > 1 else ''}):\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        local_memory.add_message(channel_id, "Bot", summary)
        return
    
    # ---- Summary command ------------------------
    if user_message.lower() == "summary":
        local_memory.add_message(channel_id, user_id, user_message)
        history = local_memory.get_chat_history(channel_id)
        summary = summarize_conversation(history)
        response = split_response(summary)
        await message.channel.send(f"📋 Summary:\n{response[0]}")
        if len(response) > 1:
            for part in response[1:]:
                await message.channel.send(part)
        local_memory.add_message(channel_id, "Bot", summary)
        return

    # ---- Query handler ---------------------
    if user_message.lower().startswith("query: "):

        user_query = user_message[7:].strip()

        if not user_query:
            await message.channel.send("❌ Please provide a query after `query:`")
            return
        
        local_memory.add_message(channel_id, user_id, user_message)

        # Pass session history (list of past queries) to query_data
        texts = query_data(user_id, user_query, session_history=local_memory.get_user_query_session_history(user_id))

        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)
        # Save current message to history as usual
        local_memory.add_message(channel_id, user_name, user_message)
        local_memory.add_message(channel_id, "Bot", str(texts))

        return

    # ---- Follow-up Query Handler (no "query:" prefix) ---------------------
    if local_memory.get_last_command_type(user_id) == "query" and not user_message.lower().startswith(("ask:", "summary", "search:", "help", "exit", "clear", "show_history", "test", "show_embeds", "gen_chat", "where_am_i")):
        user_query = user_message.strip()
        local_memory.add_message(channel_id, user_id, user_message)
        texts = query_data(user_id, user_query, session_history=local_memory.get_user_query_session_history(user_id))

        for text in texts:
            if not text.strip():
                continue
            await message.channel.send(text)

        # Save current message to history as usual
        local_memory.add_message(channel_id, user_name, user_message)
        local_memory.add_message(channel_id, "Bot", str(texts))

        return

    # ---- Existing /search handler ---------------------------------------------------------------------------------------------------------
    if user_message.lower().startswith("search: "):
        terms = user_message[len("search: "):].strip()

        await message.channel.send(f"🔎 Searching for: `{terms}`")
        quick_result = search_conversation_quick(local_memory.get_vectorstore(channel_id), terms)
        await message.channel.send(f"🔎 Search result:\n{quick_result}")
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(search_conversation, terms, local_memory.get_cached_history_documents(channel_id), quick_result)

            total_result = None
            try:
                total_result = future.result(timeout=30)
                if total_result:
                    await message.channel.send(total_result)
            except concurrent.futures.TimeoutError:
                print("Long search operation timed out, using only quick response instead.")
            except Exception as e:
                await message.channel.send(f"❌ Error: {e}")

        local_memory.clear_cached_history(channel_id)

        if total_result:
            local_memory.add_message(channel_id, user_name, user_message)
            local_memory.add_message(channel_id, "Bot", total_result if total_result else quick_result)

        return

    # ---- Testing utilities ----------------------------------------------------------------------------------------------------------------
    if user_message.lower() == "test":
        await message.channel.send(f"Your ID: {user_id}")
        await message.channel.send(f"{user_name} just sent me a message")
        return

    # Show embeds might need work on because i'm not sure how to display the embeddings in a readable format - kyle
    if user_message.lower() == "show_embeds":
        embeddings = local_memory.get_chat_embeddings(channel_id)
        print(embeddings)
        response = split_response(str(embeddings), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    # show the current chat history
    if user_message.lower() == "show_history":
        chat_history = local_memory.get_chat_history(channel_id)
        print(chat_history)
        response = split_response(str(chat_history), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    # show the current channel ID
    if user_message.lower() == "where_am_i":
        await message.channel.send(f"Channel: {message.channel.name} ({channel_id})")
        return
    
    # generate a pre made chat
    if user_message.lower() == "gen_chat":
        new_memory = [('<@458754358004416524>', 'how is the project coming along?', datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('Bot', "It's going quite well, actually! We've made excellent progress on the new data pipeline integration, even hitting a few of our stretch goals. How's the client-facing dashboard development coming along for you?", datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('<@458754358004416524>', 'the dashboard is going great as of now. we just finished up some of the nicer UI elements on it as well. The clients are loving it so far.', datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('Bot', "That's fantastic news about the UI elements! It's always great when the clients are actively engaged and happy. Are you still on track to integrate the real-time analytics module by end of next week, or has the UI work shifted that timeline at all?", datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('<@458754358004416524>', 'currently the real-time analytics are going through some problems with the database. we need to get some more permissions from Mr. Bob about that though. How is your team doing with the presentation for tomorrow?', datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('Bot', "Oh, the presentation for tomorrow is looking solid! We just wrapped up the final review of the Q3 projections and the new workflow diagrams. Everything's synced up. Did you manage to get those updated user testimonials from marketing for the demo section?", datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('<@458754358004416524>', 'the demo is awesome right now. its all interactive and has been tested thoroughly.', datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('Bot', "That's incredible to hear about the demo! Interactive and thoroughly tested is exactly what we need. So, are we still planning to showcase the new dynamic filtering capabilities within that interactive demo, or have you focused more on the core user journeys for this iteration?", datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('<@458754358004416524>', 'yes we have been talking back and forth with the client. They are changing some stuff up on us though. because of that make sure that you scrap that last slide over the current backend page for the videos. We are changing that entirely but have no clue to what as of now.', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('Bot', 'Understood on scrapping that slide for the backend video page – definitely noted. Thanks for the heads-up on the client changes, that sounds like it could be a significant pivot. Is this likely to affect the proposed integration of the secure asset management module, or is that still on track?', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('<@458754358004416524>', "everything seems to still be on-track. But i'm not sure how we want to go about figuring this all out before the demo. Do you have anyone you can ask about it to try getting an answer faster?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('Bot', "That's a good point about the demo timeline. For figuring out the new direction quickly, I'd suggest reaching out to Alex in Product Strategy. He's usually got a great bird's-eye view of client needs and can often provide a solid temporary placeholder or a quick path forward. Have you already looped in the QA team about the potential impact of these changes on their final testing pass?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('<@458754358004416524>', "Thats true! Alex and Steve can for sure help us out with this. They can reach out and get us more of an idea on what we are supposed to do next regarding the demo. As for testing I think that we should try getting someone on John's testing team to figure that out.", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('Bot', "Good call on Alex and Steve – they'll definitely help clarify the demo direction. And getting someone from John's team on the testing impact is smart. Have you thought about how we'll communicate these potential last-minute changes to the training materials team, especially if the demo flow shifts significantly?", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('<@458754358004416524>', "I've just sent them a message about this topic. Steve says that they are down to help us out!", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('Bot', "That's excellent news about Steve and the training materials team! Glad they're on board to help. Does this mean we'll need to re-evaluate the integration points for the new user onboarding flow, given the potential demo changes?", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('<@458754358004416524>', 'honestly i think everything is looking good for now. ill call you later. Goodbye!', datetime.datetime(2025, 6, 19, 19, 51, 17, 303753)), ('Bot', "Sounds good! Glad to hear everything's shaping up. Talk later then! Just to confirm for my planning, are we still on track to get the final sign-off on the API documentation updates by end of day today?", datetime.datetime(2025, 6, 19, 19, 51, 17, 303753))]
        local_memory.set_chat_history(channel_id, new_memory)
        
        await message.channel.send("✅🕒 Memory cleared and generated with time!")
        return

    # clear the bot's memory
    if user_message.lower() == "clear":
        local_memory.clear_chat_history(channel_id)
        await message.channel.send("🧠 Memory cleared!")
        return
    
    # terminates the bot
    if user_message.lower() == "exit":
        await message.channel.send("🔒 Saving memory and shutting down...")
        local_memory.store_all_in_long_term_memory()
        await message.channel.send("💀 Goodbye!")
        exit()

    # ---- Help command -----------------------------------------------------------------------------------------------------------------------
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
            "Use `test` to test the bot's response.\n"
            "Use `show_history` to display the current conversation history.\n"
            "Use `where_am_i` to find out which channel you are in.\n"
            "Use `help` to see this message again.\n"
        )
        return

    # Only allow default chat if message starts with 'ask: '
    if user_message.lower().startswith("ask: "):
        local_memory.add_message(channel_id, user_id, user_message)

        # Build prompt
        
        full_prompt = f"""
        Be Formal with your replies. This is a work environment.
        I do NOT want any bullet points when you respond to me unless I ask you for them.
        These replies MUST be short as to not clutter the text chat.
        All replies must be readable. Now respond to this given the previous instructions:\n
        """

        # Test Data Creation Prompt:
        # full_prompt = "You are my coworker and we are having a conversation about one of our projects. make up details about it when i ask you something. never say you dont know something, always answer. also come up with fake follow up questions for me if possible. This conversation will be used as training data, so do not worry about accuracy. Make sure to keep your responses limited in length as to not clutter the text chat. Now using these instructions reply to me:\n"

        for role, msg, _ in local_memory.get_chat_history(channel_id):
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
            
        local_memory.add_message(channel_id, "Bot", bot_reply)
        return
    else:
        local_memory.add_message(channel_id, user_name, user_message)

client.run(discord_token)
