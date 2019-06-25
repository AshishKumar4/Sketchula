#!/usr/bin/python3
from flask import *
from flask_restful import *
from json import dumps
import subprocess

from Engines import *

import numpy as np
import matplotlib.pyplot as plt
import pickle 
import numpy as np

app = Flask(__name__)
api = Api(app)

SketchModelMap = dict()

StyleTransferModel = StyleTransfer_Engine()

userModelRoot = lambda obj: './models/'# + obj['user']

getModelLocation = {'private':lambda a:"./models/private/"+a, 'vanilla':lambda a:"./models/vanilla/"}

class Generate(Resource):
    def post(self):
        # Get Data (Image/Vector), type (private/vanilla), 'user' - username, modelname
        obj = request.get_json(force=True)
        print(type(obj['data']))
        model_id = hash(obj['user'] + obj['modelname'])
        if model_id not in SketchModelMap:
            print("Got here")
            model_location = getModelLocation[obj['type']](obj['user'])
            print(model_location)
            model = MagicBox()
            model.load_networks(save_dir=model_location, name=obj['modelname'])
            SketchModelMap[model_id] = model
        print("Unpacking Image")
        image = pickle.loads(bytes.fromhex(obj['data']))
        print(type(image))
        print('Feeding in Image')
        results, status = SketchModelMap[model_id].run(image, transformed=obj['transformed'])
        print(type(results))
        return jsonify(pickle.dumps(results).hex())

class StyleTransfer(Resource):
    def post(self):
        # Get Data (Image/Vector), type (private/vanilla), 'user' - username, modelname
        obj = request.get_json(force=True)
        print("Unpacking Image")
        image = pickle.loads(bytes.fromhex(obj['data']))
        style = pickle.loads(bytes.fromhex(obj['style']))
        print(type(image))
        print(type(style))
        #print(obj)
        results, status = StyleTransferModel.run(image, style)
        print(type(results))
        results.save(str(np.random.randint(1000000)) + ".png", "png")
        return jsonify(pickle.dumps(results).hex())
    
class Train(Resource):
    def post(self):
        # Get Data (Image/Vector), labels (Names), type (img,vec), 'user' - username
        obj = request.get_json(force=True)
        #print(obj)
        #print(results)
        return jsonify(results)

class loadModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']] =  MagicBox(userModelRoot(obj))
        return jsonify("Loaded")

class saveModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']].save(userModelRoot(obj))
        return jsonify("Saved")

class createModel(Resource):
    def post(self):
        obj = request.get_json(force=True)
        privateModels[obj['user']] = MagicBox(userModelRoot(obj), create=True)
        return jsonify("Created")

api.add_resource(Generate, '/generate')
api.add_resource(StyleTransfer, '/transfer')

api.add_resource(Train, '/train')

api.add_resource(saveModel, '/save')
api.add_resource(loadModel, '/load')

api.add_resource(createModel, '/create')

# WARNING: DO NOT RUN THIS ON MULTITHREADED WSGI SERVERS!
if __name__ == '__main__':
     app.run(host='0.0.0.0', port='5000')



