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
    Creates tables 'session', 'message' and 'behavior' if they do not already exist.

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