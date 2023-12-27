import sqlite3


def main():
    """
    This function/file does the following:

    - creates/opens a sqlite database to store our backend data
    - creates a table for highlights and books
    - fetches highlights from readwise in chunks and stores them in the sqlite database
    - fetches books from readwise based on what is stored in the sqlite database and stores them in the database
    :return:
    """
    conn = sqlite3.connect('babotree.db')

if __name__ == '__main__':
    main()