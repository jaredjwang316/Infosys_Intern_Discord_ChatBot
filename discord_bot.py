"""
discord_bot.py

Purpose:
--------
This script launches an AI-powered chatbot as a Discord bot using Python. The bot responds to user messages 
intelligently by leveraging Google’s Gemini model through LangChain and LangGraph, with full memory and 
tool-using capabilities. 

Rather than building a language model from scratch, this project integrates Gemini via API to power natural 
language responses, enabling a lightweight, intelligent assistant experience directly inside Discord.

Key Technologies:
-----------------
- 🧠 **Gemini API (via LangChain)** — Handles language understanding and response generation.
- 🔗 **LangGraph + LangChain** — Powers decision-making logic and tool integration (e.g., search, summarization).
- 💾 **LocalMemory** — Manages per-channel memory for context-aware conversations, aka short-term memory.
- 💬 **discord_bot.py** — Handles real-time interaction with users in Discord servers.
"""
import os
import discord
import logging
import sys
import asyncio
import io

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import datetime

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from memory_storage import memory_storage
from agent import Agent

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

# ── Discord client setup ──────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def split_response(response, line_split=True):
    """
    Splits a long string into smaller parts that comply with Discord's 1900-character message limit.

    Parameters:
        response (str): The original message string.
        line_split (bool): If True, split by line breaks to preserve readability. Otherwise, split by length.

    Returns:
        list[str]: A list of smaller message strings to send.
    """
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
    """
    Discord event triggered when the bot has successfully connected.
    Sends a startup message to the first available text channel in each guild.
    """
    print(f"Logged in as {client.user}")
    # Send a message to all text channels the bot can access
    for guild in client.guilds:
        for channel in guild.text_channels:
            try:
                await channel.send("🤖 I am up and ready!\n For a full list of commands type: `help`")
                break  # Only send to the first accessible text channel per guild
            except Exception:
                continue
            
def get_user_permission_role(member_roles: list) -> str:
    """
    Determines the simplified permission role of a user based on their Discord roles.

    Args:
        member_roles (list): A list of discord.Role objects for the user.

    Returns:
        str: "administrator", "supervisor", "member", or "none".
    """
    user_discord_role_names = {role.name for role in member_roles}

    if "Administrator" in user_discord_role_names:
        return "administrator"
    elif "Supervisor" in user_discord_role_names:
        return "supervisor"
    elif "Member" in user_discord_role_names:
        return "member"
    else:
        return "none" # This is the key part for "non-member/supervisor/admin"
    
@client.event
async def on_message(message):
    """
    Discord event triggered when a message is sent in a text channel.

    Parameters:
        message (discord.Message): The message object sent by a user.
    """

    if message.author == client.user: #discord bot ignores its own messages
        return
    
    logging.info(f"📨 Message received — User: {message.author} (ID: {message.author.id})")
    logging.info(f"📨 Channel: {message.channel} (ID: {message.channel.id})")
    logging.info(f"💬 Content: {message.content}")

    user_id      = message.author.id    #unique Discord user ID of the message sender.
    user_name    = message.author.mention   #string to mention/tag the user in a message.
    user_message = message.content.strip()  #actual text content of the message the user sent.
    channel_id   = message.channel.id   #The unique ID of the channel where the message was sent.
    now = datetime.datetime.utcnow()

    user_permission_role = get_user_permission_role(message.author.roles)
    
    if user_permission_role == "none":
        # Log the unauthorized attempt, but DO NOT send a message to the channel
        logging.warning(f"🚫 Silently ignoring unauthorized message from {message.author} (ID: {user_id}) in channel {message.channel} (ID: {channel_id}). Content: '{user_message}'")
        return # Stop further processing for unauthorized users without sending a public message
    
    # ---- Testing utilities ----------------------------------------------------------------------------------------------------------------
    if user_message.lower() == "test":
        """
        Sends back the user's Discord ID and a mention message to confirm the bot received input.
        """
        await message.channel.send(f"Your ID: {user_id}")
        await message.channel.send(f"{user_name} just sent me a message")
        return

    # Show embeds might need work on because i'm not sure how to display the embeddings in a readable format - kyle
    if user_message.lower() == "show_embeds":
        """
        Retrieves and displays the embeddings stored for the current channel.
        """
        embeddings = memory_storage.local_memory.get_chat_embeddings()
        print(embeddings)
        response = split_response(str(embeddings), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    # show the current chat history
    if user_message.lower() == "show_history":
        """
        Displays the current channel's stored message history.
        """
        chat_history = memory_storage.local_memory.get_chat_history(channel_id)
        print(chat_history)
        response = split_response(str(chat_history), line_split=False)
        for part in response:
            await message.channel.send(part)
        return

    # show the current channel ID
    if user_message.lower() == "where_am_i":
        """
        Returns the name and ID of the channel the command was issued in.
        """
        await message.channel.send(f"Channel: {message.channel.name} ({channel_id})")
        return
    
    # generate a pre made chat
    if user_message.lower() == "gen_chat":
        """
        Injects a simulated, timestamped conversation into memory for testing.
        """
        new_memory = [('<@458754358004416524>', 'how is the project coming along?', datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('Bot', "It's going quite well, actually! We've made excellent progress on the new data pipeline integration, even hitting a few of our stretch goals. How's the client-facing dashboard development coming along for you?", datetime.datetime(2025, 6, 19, 19, 43, 1, 164403)), ('<@458754358004416524>', 'the dashboard is going great as of now. we just finished up some of the nicer UI elements on it as well. The clients are loving it so far.', datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('Bot', "That's fantastic news about the UI elements! It's always great when the clients are actively engaged and happy. Are you still on track to integrate the real-time analytics module by end of next week, or has the UI work shifted that timeline at all?", datetime.datetime(2025, 6, 19, 19, 43, 37, 376236)), ('<@458754358004416524>', 'currently the real-time analytics are going through some problems with the database. we need to get some more permissions from Mr. Bob about that though. How is your team doing with the presentation for tomorrow?', datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('Bot', "Oh, the presentation for tomorrow is looking solid! We just wrapped up the final review of the Q3 projections and the new workflow diagrams. Everything's synced up. Did you manage to get those updated user testimonials from marketing for the demo section?", datetime.datetime(2025, 6, 19, 19, 44, 35, 681871)), ('<@458754358004416524>', 'the demo is awesome right now. its all interactive and has been tested thoroughly.', datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('Bot', "That's incredible to hear about the demo! Interactive and thoroughly tested is exactly what we need. So, are we still planning to showcase the new dynamic filtering capabilities within that interactive demo, or have you focused more on the core user journeys for this iteration?", datetime.datetime(2025, 6, 19, 19, 45, 21, 254802)), ('<@458754358004416524>', 'yes we have been talking back and forth with the client. They are changing some stuff up on us though. because of that make sure that you scrap that last slide over the current backend page for the videos. We are changing that entirely but have no clue to what as of now.', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('Bot', 'Understood on scrapping that slide for the backend video page – definitely noted. Thanks for the heads-up on the client changes, that sounds like it could be a significant pivot. Is this likely to affect the proposed integration of the secure asset management module, or is that still on track?', datetime.datetime(2025, 6, 19, 19, 46, 31, 496278)), ('<@458754358004416524>', "everything seems to still be on-track. But i'm not sure how we want to go about figuring this all out before the demo. Do you have anyone you can ask about it to try getting an answer faster?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('Bot', "That's a good point about the demo timeline. For figuring out the new direction quickly, I'd suggest reaching out to Alex in Product Strategy. He's usually got a great bird's-eye view of client needs and can often provide a solid temporary placeholder or a quick path forward. Have you already looped in the QA team about the potential impact of these changes on their final testing pass?", datetime.datetime(2025, 6, 19, 19, 47, 49, 602717)), ('<@458754358004416524>', "Thats true! Alex and Steve can for sure help us out with this. They can reach out and get us more of an idea on what we are supposed to do next regarding the demo. As for testing I think that we should try getting someone on John's testing team to figure that out.", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('Bot', "Good call on Alex and Steve – they'll definitely help clarify the demo direction. And getting someone from John's team on the testing impact is smart. Have you thought about how we'll communicate these potential last-minute changes to the training materials team, especially if the demo flow shifts significantly?", datetime.datetime(2025, 6, 19, 19, 49, 31, 920950)), ('<@458754358004416524>', "I've just sent them a message about this topic. Steve says that they are down to help us out!", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('Bot', "That's excellent news about Steve and the training materials team! Glad they're on board to help. Does this mean we'll need to re-evaluate the integration points for the new user onboarding flow, given the potential demo changes?", datetime.datetime(2025, 6, 19, 19, 50, 42, 193651)), ('<@458754358004416524>', 'honestly i think everything is looking good for now. ill call you later. Goodbye!', datetime.datetime(2025, 6, 19, 19, 51, 17, 303753)), ('Bot', "Sounds good! Glad to hear everything's shaping up. Talk later then! Just to confirm for my planning, are we still on track to get the final sign-off on the API documentation updates by end of day today?", datetime.datetime(2025, 6, 19, 19, 51, 17, 303753))]
        memory_storage.local_memory.set_chat_history(channel_id, new_memory)

        await message.channel.send("✅🕒 Memory cleared and generated with time!")
        return

    # clear the bot's memory
    if user_message.lower() == "clear":
        """
        Clears all stored messages from current channel’s memory, aka short-term memory.
        """
        memory_storage.local_memory.clear_chat_history(channel_id)
        await message.channel.send("🧠 Memory cleared!")
        return
    
    # terminates the bot
    if user_message.lower() == "exit":
        """
        Gracefully shuts down the bot and saves all in-memory data.
        """
        await message.channel.send("🔒 Saving memory and shutting down...")
        memory_storage.store_all_in_long_term_memory()
        await message.channel.send("💀 Goodbye!")
        await client.close()  # Properly closes all tasks

        # Delay exit to give `client.close()` time to propagate
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(loop.stop)
        return  # prevent further handling

    # ---- Help command -----------------------------------------------------------------------------------------------------------------------
    if user_message.lower() == "help":
        """
        Sends a help message listing all available commands.
        """
        await message.channel.send(
            "# General Help:\n"
            "Use `ask: <your question>` to ask the bot a question.\n"
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

    
    if user_message.lower().startswith("ask: "):
        """
        Sends the user's message to the LangChain agent and returns the bot's response.

        Expected input: `ask: <your message>`
        """
        messages = HumanMessage(content=user_message)

        config = memory_storage.get_config(channel_id)

        ### ROLE BASED ACCESS #####################################################################
        if user_permission_role == "administrator":
            # Call agent for admin
            allowed_tools = ['query', 'summarize', 'summarize_by_time', 'search'] # Tools allowed for admin use
            role_name = "admin_agent"
            agent = Agent(role_name=role_name, allowed_tools=allowed_tools)
            logging.info(f"User {user_name} (Admin) is using agent with tools: {allowed_tools}")

            response = agent.invoke({
                "current_channel": channel_id,
                "current_user": user_id,
                "messages": [messages]
        }, config)
        elif user_permission_role == "supervisor":
            # Call agent for supervisor
            allowed_tools = ['query', 'summarize', 'summarize_by_time', 'search'] # Tools allowed for supervisor use
            role_name = "supervisor_agent"
            agent = Agent(role_name=role_name, allowed_tools=allowed_tools)
            logging.info(f"User {user_name} (Supervisor) is using agent with tools: {allowed_tools}")

            response = agent.invoke({
                "current_channel": channel_id,
                "current_user": user_id,
                "messages": [messages]
        }, config)
            
        elif user_permission_role == "member":
            # Call agent for members
            allowed_tools = ['summarize', 'summarize_by_time', 'search'] # Tools allowed for member use MEMBERS CANNOT QUERY
            role_name = "member_agent"
            agent = Agent(role_name=role_name, allowed_tools=allowed_tools)
            logging.info(f"User {user_name} (Member) is using agent with tools: {allowed_tools}")

            response = agent.invoke({
                "current_channel": channel_id,
                "current_user": user_id,
                "messages": [messages]
        }, config)
        else:
            # This case should ideally not be reached due to the initial 'none' check,
            # but it's good practice to have a fallback or raise an error.
            # For robustness, we'll assign a very limited set or log an unexpected state.
            allowed_tools = [] # No tools if somehow this path is hit for an unauthorized user
            logging.error(f"Unexpected: User {user_name} with role '{user_permission_role}' reached agent invocation. Assigning no tools.")

        ##########################################################################################################################################
        
        try:
            bot_reply = response["messages"][-1].content.strip()
            
            formatted_reply = ""
            if "TASK:" in bot_reply:
                task_start = bot_reply.find("TASK:") + len("TASK:")
                task_end = bot_reply.find("\n", task_start)
                if task_end == -1:
                    task_end = len(bot_reply)
                formatted_reply = bot_reply[task_start:task_end].strip()
            else:
                formatted_reply = bot_reply
            
            print(response["messages"])

            memory_storage.add_message(channel_id, "Bot", formatted_reply)
            replies = split_response(formatted_reply)
            for reply in replies:
                await message.channel.send(reply)

            images = response.get("images", [])
            if images:
                logging.info(f"📸 Sending {len(images)} images in response to {user_name} in channel {channel_id}")
                for filename, file_data in images:
                    try:
                        if isinstance(file_data, bytes):
                            file_buffer = io.BytesIO(file_data)
                        else:
                            file_buffer = file_data
                        discord_file = discord.File(file_buffer, filename=filename)
                        await message.channel.send(file=discord_file)
                        logging.info(f"✅ Image {filename} sent successfully.")
                    except Exception as e:
                        logging.error(f"❌ Failed to send image {filename}: {e}")
                        await message.channel.send(f"❌ Error sending image {filename}")
        except Exception as e:
            await message.channel.send(f"❌ Error: {e}")
            return
    else:
        # Store non-command messages in local memory for future context
        memory_storage.add_message(channel_id, user_name, user_message)

try:
    client.run(discord_token)
except KeyboardInterrupt:
    print("⚙️ Bot interrupted and shutting down...")
    sys.exit(0)
except SystemExit:
    print("✅ Bot exited successfully.")
    raise