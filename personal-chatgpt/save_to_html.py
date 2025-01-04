import os
import re
import shutil
from typing import List, Dict

import markdown
from mysql.connector import Error
from pygments.formatters import HtmlFormatter
import streamlit as st


def copy_image_to_export_dir_in_docker(image_path: str):
    """
    Copy an image file to a Docker-mounted export directory.

    This function copies the specified image file to a directory that is mounted between
    the Docker container and the host system, making the image accessible from the host.

    Args:
        image_path (str): The source path of the image file to copy

    Returns:
        This function has no return.

    Raises:
        OSError: If there are permission issues creating the export directory
        shutil.Error: If the file copy operation fails
    """
    # Retrieve EXPORT_DIR value from environment variable, with a fallback
    export_dir = os.getenv('EXPORT_DIR', '/app/exported-images')

    # Ensure the exported directory exists (inside Docker)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        
    # Determine the exported path inside Docker (which is bind-mounted to host)
    exported_path = os.path.join(export_dir, os.path.basename(image_path))
    # Copy the image to the host-accessible directory
    shutil.copy(image_path, exported_path)


def convert_messages_to_markdown(messages: List[Dict[str, str]], code_block_indent='                 ') -> str:
    """
    Convert chat messages to markdown format with proper styling and image handling.

    Args:
        messages: List of message dictionaries containing 'role', 'content' and 'image' fields
        code_block_indent: String to use for indenting code blocks, defaults to 17 spaces

    Returns:
        str: Formatted markdown string with messages and images

    Example message format:
        {
            'role': 'user',
            'content': 'Hello!', 
            'image': '/path/to/image.jpg'
        }
    """
    # Check if running in Docker by checking for common Docker environment indicators
    is_docker = os.path.exists('/.dockerenv') or os.path.exists('/app')

    markdown_lines = []
    for message in messages:
        role = message['role']
        content = message['content']
        indented_content = _indent_content(content, code_block_indent)

        if message["image"] != "":
            if is_docker:
                # Copy image to exported-images directory
                copy_image_to_export_dir_in_docker(message['image'])
                # Construct the path to use in local host machine by .html file
                host_path = os.getenv('HOST_DESINATION_DIR', 'images')
                file_name = os.path.basename(message['image'])
                host_path_file_name = os.path.join(host_path, file_name)
                image_url = host_path_file_name
            else:
                # Locally, use file:// protocol with absolute path
                image_path = os.path.abspath(message['image'])
                image_url = f"file://{image_path}"

            markdown_lines.append(f"###*{role.capitalize()}*:\n![Image]({image_url})\n{indented_content}\n")

        else:
            markdown_lines.append(f"###*{role.capitalize()}*:\n{indented_content}\n")
    
    return '\n\n'.join(markdown_lines)


def _indent_content(content: str, code_block_indent: str) -> str:
    """
    Helper function to indent the content for markdown formatting.

    Args:
        content (str): The content of the message to be indented.
        code_block_indent (str): The string used to indent lines within code blocks.

    Returns:
        The indented content as a string.
    """
    if content is not None:
        lines = content.split('\n')
        indented_lines = []
        in_code_block = False  # Flag to track whether we're inside a code block

        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                indented_lines.append(line)
            elif not in_code_block:
                line = f"> {line}"
                indented_lines.append(line)
            else:
                indented_line = code_block_indent + line  # Apply indentation
                indented_lines.append(indented_line)

        return '\n'.join(indented_lines)
    
    else:
        return ""


def markdown_to_html(md_content: str) -> str:
    """
    Converts markdown content to HTML with syntax highlighting and custom styling.

    This function takes a string containing markdown-formatted text and converts it to HTML.
    It applies syntax highlighting to code blocks and custom styling to certain HTML elements.

    Args:
        md_content (str): A string containing markdown-formatted text.

    Returns:
        A string containing the HTML representation of the markdown text, including a style tag
        with CSS for syntax highlighting and custom styles for the <code> and <em> elements.
    """

    # Convert markdown to HTML with syntax highlighting
    html_content = markdown.markdown(md_content, extensions=['fenced_code', 'codehilite'])

    html_content = re.sub(
        r'<code>', 
        '<code style="background-color: #f7f7f7; color: green;">', 
        html_content)
    
    html_content = re.sub(
        r'<h3>', 
        '<h3 style="color: blue;">', 
        html_content)
    
    # Get CSS for syntax highlighting from Pygments
    css = HtmlFormatter(style='tango').get_style_defs('.codehilite')

    return f"<style>{css}</style>{html_content}"


def get_summary_and_return_as_file_name(conn, session1: int) -> str:
    """
    Retrieves the summary of a given session from the database and formats it as a file name.

    This function queries the 'session' table for the 'summary' field using the provided session ID.
    If a summary is found, it formats the summary string by replacing spaces with underscores and
    removing periods, then returns it as a potential file name.

    Args:
        conn: A connection object to the MySQL database.
        session_id (int): The ID of the session whose summary is to be retrieved.

    Returns:
        A string representing the formatted summary suitable for use as a file name, or None if no summary is found.

    Raises:
        Raises an exception if there is a failure in the database operation.
    """
    try:
        with conn.cursor() as cursor:
            sql = "SELECT summary FROM session WHERE session_id = %s"
            val = (session1, )
            cursor.execute(sql, val)

            result = cursor.fetchone()
            if result is not None and result[0] is not None:
                string = result[0].replace(" ", "_")
                string = string.replace(".", "")
                string = string.replace(",", "")
                string = string.replace('"', '')
                string = string.replace(':', '_')

                return string
            else:
                return "Acitive_current_session"

    except Error as error:
        st.error(f"Failed to get session summary: {error}")
        return ""
    

def is_valid_file_name(file_name: str) -> bool:
    """
    Checks if the provided file name is valid based on certain criteria.

    This function checks the file name against a set of rules to ensure it does not contain
    illegal characters, is not one of the reserved words, and does not exceed 255 characters in length.

    Args:
        file_name (str): The file name to validate.

    Returns:
        bool: True if the file name is valid, False otherwise.
    """
    illegal_chars = r'[\\/:"*?<>|]'
    reserved_words = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4', 
                      'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2', 'LPT3', 
                      'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']

    # Check for illegal characters, reserved words, and length constraint
    return not (re.search(illegal_chars, file_name) or
                file_name in reserved_words or
                len(file_name) > 255)