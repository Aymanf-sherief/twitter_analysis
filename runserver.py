"""
This script runs the GP_flask application using a development server.
"""

from os import environ
from GP_flask import app
import nltk
if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555


    PORT = 5555

    print(PORT)
    app.debug = True
    app.run(HOST, PORT)
