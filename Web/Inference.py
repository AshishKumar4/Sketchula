import pickle
import json
import requests as re

class InferenceWrapper:
    def __init__(self, engine):
        self.engine = engine
        #engine.__init__()
        return None 
    def generate(self,text):
        return self.engine.generate(text)
    def transfer(self,text):
        return self.engine.transfer(text)

from socket import *
import numpy as np 

class Default_InferenceEngine:
    def __init__(self):
        #self.soc = socket(AF_INET, SOCK_STREAM)
        #self.soc.connect(('127.0.0.1', 8200))
        self.mlApiRoot = "http://localhost:5000/"
        return None 
    def generate(self, obj):
        #self.soc.send(bytes(text['text'], 'utf-8'))
        #val = self.soc.recv(4096)#self.evalModel.predict(self.embed([text['text']])[:1])
        obj['transformed'] = False
        print(type(obj['data']))
        val = re.post(url = self.mlApiRoot + "generate", data = json.dumps(obj))
        #print('---->>>')
        print(val)
        if val.status_code != 200:
            return None
        #print(val.text)
        #print('<--->')
        return json.loads(val.text)
    def transfer(self, obj):
        val = re.post(url = self.mlApiRoot + "transfer", data = json.dumps(obj))
        #print('---->>>')
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        #print('<--->')
        return json.loads(val.text)
    def trainDirect(self, obj):
        val = re.post(url = self.mlApiRoot + "train", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
        return None
    def saveModel(self, obj):
        val = re.post(url = self.mlApiRoot + "save", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
    def loadModel(self, obj):
        val = re.post(url = self.mlApiRoot + "load", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
    def createModel(self, obj):
        val = re.post(url = self.mlApiRoot + "create", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)