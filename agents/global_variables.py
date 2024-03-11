# This is all global variance file
import os

try:
    with open('agents/api_qianfan_AK.txt', 'r') as f:
        os.environ["QIANFAN_AK"] = f.read()
    with open('agents/api_qianfan_SK.txt', 'r') as f:
        os.environ["QIANFAN_SK"] = f.read()
except:
    with open('api_qianfan_AK.txt', 'r') as f:
        os.environ["QIANFAN_AK"] = f.read()
    with open('api_qianfan_SK.txt', 'r') as f:
        os.environ["QIANFAN_SK"] = f.read()

Usage = 0
