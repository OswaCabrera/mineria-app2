from flask import Flask
from config import config

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    app.config.from_pyfile(“../config.py”)


    # from .api import api as api_blueprint  # We will discuss blueprints shortly as well
    # app.register_blueprint(api_blueprint, url_prefix=’/api/’)

    return app