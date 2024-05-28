from mysql.connector import Error
import streamlit as st


def init_mysql_timezone(conn):
    """
    Initializes the MySQL server's global time zone to 'America/Chicago'.

    This function connects to the MySQL server and sets the global time zone to 
    'America/Chicago'. It commits the changes and handles any potential errors that may occur
    during the process.

    Raises:
        mysql.connector.Error: If an error occurs during the connection or execution.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SET GLOBAL time_zone = 'America/Chicago';")

        conn.commit()
        # st.success("Database time zone set to US Central successfully.")

    except Error as error:
        st.error(f"Failed to set global time zone: {error}")
        raise


def init_database_tables(conn):
    """
    Creates tables 'session', 'message', 'behavior', and 'model' if they do not already exist.

    Throws an error through the Streamlit interface if the connection or 
    table creation fails.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS session
                (
                    session_id INT AUTO_INCREMENT PRIMARY KEY,
                    start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_timestamp TIMESTAMP,
                    summary TEXT
                );
            """
            )
            cursor.execute(
            """CREATE TABLE IF NOT EXISTS message
                (
                    message_id INT AUTO_INCREMENT PRIMARY KEY,
                    session_id INT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    role TEXT,
                    content MEDIUMTEXT,
                    FOREIGN KEY (session_id) REFERENCES session(session_id)
                );
            """
            )
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS behavior
                (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    choice TEXT
                );
            """
            )
            cursor.execute(
            """CREATE TABLE IF NOT EXISTS message_search
                (
                    message_id INT PRIMARY KEY,
                    session_id INT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    role TEXT,
                    content MEDIUMTEXT,
                    FOREIGN KEY (session_id) REFERENCES session(session_id)
                );
            """
            )
            cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model
                (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    choice TEXT
                );
            """
            )
        conn.commit()

    except Error as error:
        st.error(f"Failed to create tables: {error}")
        raise


def modify_content_column_data_type_if_different(conn):
    """
    Modifies the data type of the 'content' column in the 'message' table to MEDIUMTEXT 
    if it is not already MEDIUMTEXT. (In the early app development, the data type was originally TEXT.
    This function was made to increase the text length of the data type of the original app.) 

    Args:
        conn: A MySQL connection object.
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            SELECT DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = "chat"
            and TABLE_NAME = "message"
            AND COLUMN_NAME = "content";
            """
            cursor.execute(sql)
            result = cursor.fetchone()

            if result and result[0].upper() != "MEDIUMTEXT":
                query = """
                ALTER TABLE message
                MODIFY COLUMN content MEDIUMTEXT;
                """
                cursor.execute(query)
                conn.commit()

    except Error as error:
        st.error(f"Failed to change the column data type to MEDIUMTEXT: {error}")
        raise


def check_if_column_model_exist_in_message_table(conn) -> str:
    """ 
    This function checks if the column 'model' exists in the 'message' table in the 
    'chat' database.

    It uses a MySQL query to count the number of columns with the name 'model' in the 
    'message' table. If the count is 1, it means the column exists, and the function returns 
    'Exist'. If the count is 0, it means the column does not exist, and the function returns 
    'Not Exist'.

    Parameters: conn (Connection): A MySQL connection object.

    Returns: str: 'Exist' if the column 'model' exists in the 'message' table, 'Not Exist' 
    otherwise. 
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            SELECT IF(count(*) = 1, 'Exist','Not Exist') AS result
            FROM information_schema.columns
            WHERE
                table_schema = 'chat'
                AND table_name = 'message'
                AND column_name = 'model';
            """
            )
            result = cursor.fetchone()
            return result[0]

    except Error as error:
        st.error(f"Failed to get return whether coulum model exists in table message: {error}")
        raise


def add_column_model_to_message_table(conn):
    """
    This function checks if the column 'model' exists in the 'message' table in the 'chat' 
    database.

    If the column does not exist, it adds a new column 'model' to the 'message' table with 
    a default value of an empty string.

    Returns:
    bool: True if the column 'model' exists in the 'message' table, False otherwise.
    """
    if check_if_column_model_exist_in_message_table(conn) == 'Not Exist':
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                """
                ALTER TABLE message
                ADD COLUMN model VARCHAR(50) NOT NULL DEFAULT "" AFTER role;
                """
                )
                result = cursor.fetchone()
                if result is not None and result[0] is not None:
                    st.session_state.session = result[0]
                else:
                    st.session_state.session = None

        except Error as error:
            st.error(f"Failed to add column model in message table: {error}")
            raise
    else:
        pass


def check_if_column_model_exist_in_message_search_table(conn) -> str:
    """ 
    This function checks if the column 'model' exists in the 'message_search' table in the 
    'chat' database.

    It uses a MySQL query to count the number of columns with the name 'model' in the 
    'message' table. If the count is 1, it means the column exists, and the function returns 
    'Exist'. If the count is 0, it means the column does not exist, and the function returns 
    'Not Exist'.

    Parameters: conn (Connection): A MySQL connection object.

    Returns: str: 'Exist' if the column 'model' exists in the 'message' table, 'Not Exist' 
    otherwise. 
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            SELECT IF(count(*) = 1, 'Exist','Not Exist') AS result
            FROM information_schema.columns
            WHERE
                table_schema = 'chat'
                AND table_name = 'message_search'
                AND column_name = 'model';
            """
            )
            result = cursor.fetchone()
            return result[0]

    except Error as error:
        st.error(f"Failed to get return whether coulum model exists in table message_search: {error}")
        raise


def add_column_model_to_message_search_table(conn):
    """
    This function checks if the column 'model' exists in the 'message_search' table in 
    the 'chat' database.

    If the column does not exist, it adds a new column 'model' to the 'message_search' 
    table with a default value of an empty string.

    Returns:
    bool: True if the column 'model' exists in the 'message_search' table, False otherwise.
    """
    if check_if_column_model_exist_in_message_search_table(conn) == 'Not Exist':
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                """
                ALTER TABLE message_search
                ADD COLUMN model VARCHAR(50) NOT NULL DEFAULT "" AFTER role;
                """
                )
                result = cursor.fetchone()
                if result is not None and result[0] is not None:
                    st.session_state.session = result[0]
                else:
                    st.session_state.session = None

        except Error as error:
            st.error(f"Failed to add column model in message table: {error}")
            raise
    else:
        pass


def check_if_column_content_in_message_table_is_indexed(conn) -> str:
    """ 
    Checks if the 'content' column in the 'message' table is indexed.

    Args:
        conn: A database connection object.

    Returns:
        str: "yes" if the 'content' column is indexed, "no" otherwise.

    Raises:
        Error: If there's an error executing the SQL query.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_NAME = 'message';
            """
            )
            if "content" in [row[0] for row in cursor.fetchall()]:
                return "Yes"
            else:
                return "No"

    except Error as error:
        st.error(f"Failed to show index of the table message: {error}")
        raise


def index_column_content_in_table_message(conn):
    """
    Creates a full-text index on the 'content' column in the 'message' table if it doesn't already exist.

    Args:
        conn: A database connection object.

    Returns:
        None

    Raises:
        Error: If there's an error executing the SQL query to create the index.
    """
    if check_if_column_content_in_message_table_is_indexed(conn) == 'No':
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                """
                ALTER TABLE message
                ADD FULLTEXT(content);
                """
                )
            conn.commit()

        except Error as error:
            st.error(f"Failed to index column content in table message: {error}")
            raise
    else:
        pass
