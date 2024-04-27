import streamlit as st
from mysql.connector import connect, Error


def insert_initial_default_model_behavior(conn, behavior1: str) -> None:
    """
    Inserts the initial default model behavior into the 'behavior' table if it does not already exist.

    This function attempts to insert a new row into the 'behavior' table with the provided choice.
    The insertion will only occur if the table is currently empty, ensuring that the default
    behavior is set only once.

    Args:
        conn: A connection object to the database.
        behavior1: A string representing the default behavior to be inserted.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = """
            INSERT INTO behavior (choice) 
            SELECT %s FROM DUAL 
            WHERE NOT EXISTS (SELECT * FROM behavior);
            """
            val = (behavior1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save the initial default model behavior: {error}")
        raise


def Load_the_last_saved_model_behavior(conn) -> None:
    """
    Retrieves the last saved model behavior from the 'behavior' table in the MySQL database and
    save it to session state.

    This function selects the most recent 'choice' entry from the 'behavior' table based on the highest ID.

    Args:
        conn: A connection object to the MySQL database.

    Returns:
        The last saved model behavior as a string, or None if no behavior is found.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT choice FROM behavior ORDER BY id DESC LIMIT 1;")

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                st.session_state.behavior = result[0]
            else:
                st.session_state.behavior = None

    except Error as error:
        st.error(f"Failed to read the last saved behavior: {error}")
        raise


def return_behavior_index(behavior1: str) -> int:
    """
    Returns the index of a given behavior from a predefined dictionary of behaviors.

    This function maps a behavior description to its corresponding index based on a predefined
    dictionary of behavior descriptions and their associated indices.

    Args:
        behavior (str): A string representing the behavior description.

    Returns:
        An integer representing the index of the behavior.

    Raises:
        KeyError: If the behavior description is not found in the predefined dictionary.
    """
    behavior_dic = {
        'Deterministic (T=0.0, top_p=0.2)': 0,
        'Conservative (T=0.3, top_p=0.4)': 1,
        'Balanced (T=0.6, top_p=0.6)': 2,
        'Diverse (T=0.8, top_p=0.8)': 3, 
        'Creative (T=1.0, top_p=1.0)': 4
    }

    if behavior1 not in behavior_dic:
        raise KeyError(f"Behavior '{behavior1}' not found in the behavior dictionary.")
    
    return behavior_dic[behavior1]


def return_temp_and_top_p_values_from_model_behavior(behavior1: str) -> tuple[float, float]:
    """
    Returns the temperature and top_p values associated with a given model behavior.

    This function maps a behavior description to its corresponding temperature (T) and top_p values.

    Args:
        behavior (str): A string representing the behavior description.

    Returns:
        A tuple containing the temperature (float) and top_p (float) values.

    Raises:
        ValueError: If the behavior description is not found in the predefined list of behaviors.
    """

    behavior_to_values = {
        'Deterministic (T=0.0, top_p=0.2)': (0.0, 0.2),
        'Conservative (T=0.3, top_p=0.4)': (0.3, 0.4),
        'Balanced (T=0.6, top_p=0.6)': (0.6, 0.6),
        'Diverse (T=0.8, top_p=0.8)': (0.8, 0.8),
        'Creative (T=1.0, top_p=1.0)': (1.0, 1.0)
    }

    if behavior1 not in behavior_to_values:
        raise ValueError(f"Model behavior '{behavior1}' is not in the predefined list.")

    return behavior_to_values[behavior1]


def save_model_behavior_to_mysql(conn, behavior1: str) -> None:
    """
    Saves a model behavior to the 'behavior' table in the MySQL database.

    This function inserts a new row into the 'behavior' table with the provided choice.

    Args:
        conn: A connection object to the MySQL database.
        behavior1: A string representing the behavior to be saved.

    Raises:
        Raises an exception if the database operation fails.
    """
    try:
        with conn.cursor() as cursor:
            sql = "INSERT INTO behavior (choice) VALUE (%s)"
            val = (behavior1, )
            cursor.execute(sql, val)
            conn.commit()

    except Error as error:
        st.error(f"Failed to save model behavior: {error}")
        raise


if __name__ == "__main__":
    connection = connect(**st.secrets["mysql"])  # Get LOCAL database credentials from .streamlit/secrets.toml for development.

    # The following code handles model behavior. 
    # The behavior chosen last time will be reused rather than using a default value. 
    if "behavior" not in st.session_state:
        st.session_state.behavior = ""

    # If the behavior table is empty:
    insert_initial_default_model_behavior(connection, 'Deterministic (T=0.0, top_p=0.2)')
        
    Load_the_last_saved_model_behavior(connection)  # load from database and save to session_state
    (temperature, top_p) = return_temp_and_top_p_values_from_model_behavior(st.session_state.behavior)
    behavior_index = return_behavior_index(st.session_state.behavior)  # from string to int (0 to 4)

    behavior = st.sidebar.selectbox(
        label="Select the behavior of your model",
        placeholder='Pick a behavior',
        options=['Deterministic (T=0.0, top_p=0.2)', 'Conservative (T=0.3, top_p=0.4)', 
                'Balanced (T=0.6, top_p=0.6)', 'Diverse (T=0.8, top_p=0.8)', 'Creative (T=1.0, top_p=1.0)'],
        index=behavior_index,
        key="behavior1"
        )

    if behavior != st.session_state.behavior:  # only save to database if behavior is newly clicked 
        save_model_behavior_to_mysql(connection, behavior)

    connection.close()