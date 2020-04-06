# -*- coding:utf-8 -*-
import codecs
import os

__version__ = codecs.open(os.path.join(
    os.path.dirname(__file__), 'VERSION.txt')).read().strip()
__author__ = 'liuwen'
