#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music Li.  yuezhanl@andrew.cmu.edu

import logging
from termcolor import colored
import sys
logging.basicConfig(format=colored('[%(asctime)s @%(filename)s:%(lineno)d] %(message)s', 'green'))
logger = logging.getLogger('nnmusic')
logger.propagate = False
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(colored('[%(asctime)s @%(filename)s:%(lineno)d] %(message)s', 'green')))
logger.addHandler(handler)