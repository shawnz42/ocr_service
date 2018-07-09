#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : '__init__.py'.py 
Description:
Author: 'zhengyang' 
Date: '2017/12/15' '14:04'
"""
# flask
from flask import Flask
from flask_restful import Api


# views
from scripts.views import text_ocr_entry, TextOcr


def create_app():
    app = Flask(__name__)

    app.register_blueprint(text_ocr_entry)

    view = Api(app)
    view.add_resource(TextOcr, '/text_ocr')

    return app

app = create_app()


@app.errorhandler(404)
def page_not_found(error):
    return "please check url!"