#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
File Name : 'manage'.py 
Description:
Author: 'zhengyang' 
Date: '2018/1/3' '10:17'
"""


from scripts import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9435, debug=True)
