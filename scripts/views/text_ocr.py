#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : 'chinese_ocr'.py 
Description:
Date: '2018/1/3' '10:08'
"""


import json

import tensorflow as tf
from flask import Blueprint, request, jsonify
from flask_restful import Resource

from scripts.utils.comm_log import logger
from scripts.models.cnn_ctc import main as get_ocr_result


text_ocr_entry = Blueprint('chinese_ocr', __name__)

invalid_params_resp = {"code": False, "message": "please check params or post json"}


class TextOcr(Resource):

    def post(self):

        type_ = request.args.get('type')
        data = request.get_json()

        logger.info('type: {} data:{} '.format(type_, data))


        img_bs64 = data.get("img_bs64")

        if type_ == 'cnocr' and img_bs64:

            result = get_ocr_result(img_bs64)
            return jsonify({"code": True,
                            "res": result})
        else:
            return jsonify(invalid_params_resp)

    def get(self):
        return jsonify({"code": True, "message": "welcome to text orc"})