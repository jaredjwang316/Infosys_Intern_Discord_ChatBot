# Discord ChatBot

A sophisticated Discord bot powered by Google's Gemini AI that provides intelligent conversation, database querying, conversation search, and summarization capabilities.

## Project Overview

This Discord chatbot integrates with MySQL databases and uses advanced AI features including:
- Natural language conversation with context awareness
- SQL query generation and execution
- Conversation search using vector embeddings
- Automatic conversation summarization
- Time-based conversation filtering

## Files Structure

### Core Files

## discord_bot.py
This is the main Discord bot file that handles all Discord interactions. It contains various tools and commands that can be triggered by specific keywords and provides:
- Message handling with user and channel-specific memory
- Command processing for queries, search, and summaries
- Integration with all function modules
- Test utilities for development

## setup_database.py
A comprehensive database setup and seeding script that:
- Creates the MySQL database if it doesn't exist
- Executes the database schema from `Schema_test.sql`
- Seeds the database with realistic test data including:
  - 20 employees with various roles
  - 5 clients across different industries
  - 10 projects with assignments
  - Skills and employee-skill relationships
- Uses the Faker library to generate realistic sample data

## requirements.txt
Contains all Python dependencies needed for the project:
- Discord.py for Discord bot functionality
- LangChain ecosystem for AI and embedding features
- Google Generative AI for Gemini integration
- MySQL connector for database operations
- Vector storage libraries (pgVector)
- Additional utilities (python-dotenv, faker)

### Database Structure

## database/Schema_test.sql
Defines the complete MySQL database schema with six main tables:
- `employees` - Infosys staff information
- `clients` - External clients served by Infosys
- `projects` - Client projects managed by Infosys
- `employee_project_assignments` - Many-to-many relationship between employees and projects
- `skills` - Catalog of technical skills
- `employee_skills` - Many-to-many relationship between employees and their skills

## database/schema.txt
Auto-generated formatted version of the SQL schema used by the query function for AI context.

### Function Modules

## functions/query.py
Advanced database querying module that:
- Connects to MySQL database using environment variables
- Uses Google's Gemini AI to convert natural language to SQL
- Validates SQL queries against allowed tables for security
- Maintains conversation history for context-aware responses
- Handles database errors gracefully
- Provides intelligent responses based on query results

## functions/search.py
Conversation search functionality using vector embeddings:
- Converts conversation history into searchable documents
- Uses Google's text-embedding-004 model for embeddings
- Implemented semantic search using pgVector with cosine similarity
- Supports both timestamped and non-timestamped conversation formats
- Returns relevant conversation snippets based on search queries

## functions/summary.py
Intelligent conversation summarization:
- Processes conversation history with flexible timestamp support
- Uses Gemini AI to generate structured summaries
- Formats output with headers and bullet points
- Handles both user-bot conversations with proper role identification
- Supports time-based conversation filtering

## Bot Commands

The Discord bot responds to several specific commands:

### Basic Commands
- `help` - Display all available commands
- `clear` - Clear conversation memory
- `exit` - Stop the bot

### AI Features
- `ask: <message>` - Have a natural conversation with the bot
- `query: <question>` - Ask questions about the database (employees, projects, clients, skills)
- `summary` - Get a summary of the entire conversation
- `summary: last X [minute|hour|day|week|month|year]` - Get time-limited summary
- `search: <terms>` - Search for specific terms in conversation history

### Development/Testing
- `test` - Display user ID and basic info
- `show_history` - Display current conversation history
- `where_am_i` - Show current channel information
- `gen_time` - Generate sample conversation with timestamps
- `gen_no_time` - Generate sample conversation without timestamps

## Setup Instructions

1. **Environment Variables**: Create a `.env` file with:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   DISCORD_BOT_TOKEN=your_discord_bot_token
   MODEL_NAME=gemini-1.5-flash
   DB_HOST=<endpoint of your database instance>
   DB_PORT=5432 
   DB_USER=<master username>
   DB_PASSWORD=<master password>
   DB_NAME="postgres"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Database**:
   ```bash
   python setup_database.py
   ```

4. **Run the Bot**:
   ```bash
   python discord_bot.py
   ```

## Technical Features

- **Memory Management**: Separate conversation histories per user and channel
- **Response Splitting**: Automatically splits long AI responses to fit Discord's message limits
- **Error Handling**: Comprehensive error handling for database and AI operations
- **Security**: SQL injection protection through query validation
- **Scalability**: Vector-based search for efficient conversation retrieval
