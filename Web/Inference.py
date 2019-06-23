import pickle
import json
import requests as re

class InferenceWrapper:
    def __init__(self, engine):
        self.engine = engine
        #engine.__init__()
        return None 
    def classify(self,text):
        return self.engine.classify(text)

class AWS_InferenceEngine:
    def __init__(self):
        return None 
    def classify(self,text):
        return "Done"

class Google_InferenceEngine:
    def __init__(self):
        return None 
    def classify(self,text):
        return "Done"


import Algorithmia

class Algorithmia_InferenceEngine:
    def __init__(self):
        self.client = Algorithmia.client('simwMUzTtxRs30OR13kELwpYPnM1')
        self.algo = self.client.algo('nlp/SocialSentimentAnalysis/0.1.4')
        self.algo.set_options(timeout=300) # optional
        return None 
    def classify(self,text):
        res = self.algo.pipe(text).result
        print(res)
        return res[0]

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class vader_InferenceEngine:
    def __init__(self):
        self.model = SentimentIntensityAnalyzer()
        return None 
    def classify(self,text):
        d = self.model.polarity_scores(text['text'])
        return d

#import tensorflow_hub as hub


class Custom_NNLM__DirectInferenceEngine:
    def __init__(self):
        self.embed_method = 'nnlm128'
        #self.embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
        #self.evalModel = tf.keras.models.load_model('nnlm_evalModel.h5')
        #self.sentDict = {0:{"Negative", "Red"}, 1:{"Positive", "Green"}}
        return None 
    def classify(self,text):
        #val = self.evalModel.predict(self.embed([text['text']])[:1])
        #print(val)
        #return sentDict[np.argmax(val)]
        return None

from socket import *
import numpy as np 

class NNLM_InferenceEngine:
    def __init__(self):
        #self.soc = socket(AF_INET, SOCK_STREAM)
        #self.soc.connect(('127.0.0.1', 8200))
        self.sentDict = {'neg':{"Negative", "Red"}, 'pos':{"Positive", "Green"}}
        self.mlApiRoot = "http://localhost:5000/"
        self.infRoot = self.mlApiRoot + "infer/"
        self.trainRoot = self.mlApiRoot + "train/"
        self.fbRoot = self.mlApiRoot + "feedback/"
        self.saveRoot = self.mlApiRoot + "save/"
        self.createRoot = self.mlApiRoot + "create/"
        return None 
    def classify(self, obj):
        #self.soc.send(bytes(text['text'], 'utf-8'))
        #val = self.soc.recv(4096)#self.evalModel.predict(self.embed([text['text']])[:1])
        val = re.post(url = self.infRoot + "universal", data = json.dumps(obj))
        #print('---->>>')
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        #print('<--->')
        return json.loads(val.text)
    def feedback(self, obj):
        val = re.post(url = self.fbRoot + "universal", data = json.dumps(obj))
        #print('---->>>')
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        #print('<--->')
        return json.loads(val.text)
    def trainDirect(self, obj):
        val = re.post(url = self.loadRoot + "universal", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
        return None
    def saveModel(self, obj):
        val = re.post(url = self.saveRoot + "universal", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
    def loadModel(self, obj):
        val = re.post(url = self.loadRoot + "universal", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)
    def createModel(self, obj):
        val = re.post(url = self.createRoot + "universal", data = json.dumps(obj))
        print(val)
        if val.status_code != 200:
            return None
        print(val.text)
        return json.loads(val.text)