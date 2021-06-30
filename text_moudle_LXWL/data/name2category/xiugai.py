# coding: utf-8
import pandas as pd
from function import Index

with open('E:/quyushuju/shitishibie/text-classification-cnn-master/data/name2category/name2category.val.txt', 'r') as fpr:
    content = fpr.read()
content = content.replace('{', '')
print(content)
with open('E:/quyushuju/shitishibie/text-classification-cnn-master/data/name2category/name2category.val.txt', 'w') as fpw:
    fpw.write(content)