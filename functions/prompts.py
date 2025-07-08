no_result_instructions = """
        The SQL query you provided did not return any results. This may be due to an error in the query or a lack of matching data in the database. Please generate a correct SQL query based on the user's request and the database schema to find more information about the data to help you refine your query.
        
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that retrieves more information about the data.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email address.

        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Find more information about the data to help you refine your query.
        """

not_valid_instructions = """
        The SQL query you provided is not valid. Please generate a correct SQL query based on the user's request and the database schema to find more information about the data to help you refine your query.
        
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that retrieves more information about the data.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - DO NOT SHOW ID COLUMNS UNLESS SPECIFICALLY REQUESTED.
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email address.

        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Find more information about the data to help you refine your query.
        """

retry_instructions = """
        You previously generated a SQL query that did not return any results. Based on the new information retrieved, please refine your SQL query to ensure it fulfills the user's request and retrieves relevant data.
        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email address.

        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Find more information about the data to help you refine your query.
        """

reprompt_instructions = """
        The SQL query you provided is not valid. Please generate a correct SQL query that fulfills the user's request.

        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that retrieves more information about the data.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email address.
        
        Remember to:
        - Ensure the SQL query is syntactically correct.
        - Fulfill the user's request.
        """

prompt_instructions = """
        You are an expert at querying databases. Your task is to generate a SQL query based on the user's request.

        ### INSTRUCTION ###
        Given the database schema below, generate a SQL query that fulfills the user's request.
        - Ensure the SQL query is syntactically correct.
        - Use appropriate table and column names from the schema.
        - Do not use comments, markdown, or any other formatting in the SQL query (i.e. sql```...```).
        - DO NOT SHOW ID COLUMNS UNLESS SPECIFICALLY REQUESTED.
        - If asked for a certain type of email (.com, .org, etc.), search the end of the email address."""