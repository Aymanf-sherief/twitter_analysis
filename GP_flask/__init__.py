"""
The flask application package.
"""

from flask import Flask
print(__name__)
app = Flask(__name__, static_folder='D:\\GP_twitter_analysis\\GP_flask\\static')

import GP_flask.views
