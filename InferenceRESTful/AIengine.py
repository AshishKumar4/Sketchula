
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import sklearn
import re
from scipy.spatial import distance
from skimage.transform import resize
import subprocess

from sklearn.svm import SVC
from scipy import misc

import cv2
import dlib
import tensorflow as tf

import mxnet as mx
from align_dlib import *

np.random.seed(10)

import tensorflow as tf
if tf.__version__ == '2.0.0-alpha0':
    coreModel = tf.keras.models.load_model("./models/facenet_512.h5")
else:
    import keras
    coreModel = keras.models.load_model("./models/facenet_512_tf1.h5", custom_objects={'tf': tf})
DISTANCE_THRESHOLD = 0.4
final_img_size = 160
"""
from ArcFace import *
coreModel = ArcFace('./models/arcface')
DISTANCE_THRESHOLD = 0.46
final_img_size = 112
"""

def to_rgb(img):
    if img.ndim == 2:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=img.dtype)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    elif img.shape[2] == 3:
        return img
    elif img.shape[2] == 4:
        w, h, t = img.shape
        ret = np.empty((w, h, 3), dtype=img.dtype)
        # ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return img[:, :, :3]

class AIengine:
    def __init__(self, modelpath='./models', create=False):
        try:
            self.modelpath = modelpath
            classifier = modelpath + "/model.pkl"
            meta = modelpath + "/model.meta"
            self.clfMap = {'img': self.classifyImg, 'vec': self.classifyVec}
            self.fitMap = {'img': self.fitImg, 'vec': self.fitVec}
            print(classifier)
            if create:
                print("Creating new AI Engine")
                #   We Need to create this AI engine and save it on disk
                self.classifier = SVC(kernel='linear', probability=True)

                self.labelEncodeMap = dict()    # Contains mapping from id string to hash
                # self.metadata['labelDecodeMap']   # Contains mapping from hash to id string
                self.labelDecodeMap = dict()

                self.image_size = 160
                self.margin = 1.1

                subprocess.getoutput("mkdir " + modelpath)
                self.save(modelpath)
            else:
                if ("No such file or directory" in (subprocess.getoutput("ls " + modelpath))):
                    return "AI Engine not created yet!"
                print("Loading AI Engine")
                try:
                    self.classifier = pickle.loads(open(classifier, 'rb').read())
                except Exception as e:
                    print(e)
                    print("Seems like the classifier was not found/is corrupt. Would make a new one for you")
                    self.classifier = SVC(kernel='linear', probability=True)
                self.metadata = pickle.loads(open(meta, 'rb').read())

                # Contains mapping from id string to hash
                self.labelEncodeMap = dict(self.metadata['labelEncodeMap'])
                # self.metadata['labelDecodeMap']   # Contains mapping from hash to id string
                self.labelDecodeMap = {value: key for key,
                                       value in self.labelEncodeMap.items()}

                self.image_size = self.metadata['imagesize']  # 160
                self.margin = self.metadata['similarity_margin']  # 1.1
        except Exception as e:
            print("Error in AIengine.init")
            print(e)
        return None
    def embed(self, images, preprocess=False, detections=False):
        try:
            status = True
            if preprocess is True:
                images, status = self.preprocess(images, 10)
            emb = l2_normalize(coreModel.predict(images))
            return emb, status
        except Exception as e:
            print("Error in AIengine.embed")
            print(e)
        return None, False

    def fitImg(self, images, labels):
        try:
            embs, status = self.embed(images)
            self.classifier.fit(embs, labels)
            return embs
        except Exception as e:
            print("Error in AIengine.fitImg")
            print(e)
        return False

    def fitVec(self, vectors, labels):
        try:
            embs, status = vectors  # self.embed(images)
            self.classifier.fit(embs, labels)
            return embs
        except Exception as e:
            print("Error in AIengine.fitVec")
            print(e)
        return False

    def fit(self, data, labels, fitType='img'):
        try:
            lbls = list()
            for i in labels:
                if i not in self.labelEncodeMap:
                    self.labelEncodeMap[i] = hash(i)
                    self.labelDecodeMap[hash(i)] = i
                lbls.append(self.labelEncodeMap[i])
            print(self.labelEncodeMap)
            return self.fitMap[fitType](data, lbls).tolist()
        except Exception as e:
            print("Error in AIengine.fit")
            print(e)
        return False

    def classifyImg(self, images, preprocess=True):
        vectors, status = self.embed(images, preprocess)
        val = [self.labelDecodeMap[i] for i in self.classifier.predict(vectors) if i in self.labelDecodeMap]
        if len(val) == 0:
            return None, False 
        return val, status

    def classifyVec(self, vectors, preprocess):
        if vectors is None:
            return None, False
        hashs = self.classifier.predict(vectors)
        val = [self.labelDecodeMap[i] for i in hashs if i in self.labelDecodeMap]
        for i in hashs:
            if i not in self.labelDecodeMap:
                print("hash " + str(i) + " Not in map")
        if val is None or len(val) == 0:
            print("no class predicted")
            return None, False 
        return val, True

    def classify(self, data, clfType='img', preprocess=True):
        try:
            return self.clfMap[clfType](data, preprocess)
        except Exception as e:
            print(clfType)
            print("Error in AIengine.classify")
            print(e)
        return None, False

    def isSimilarII(self, img1, img2, margin=1.1):
        try:
            v1 = self.embed([img1])
            v2 = self.embed([img2])
            dis = distance.cosine(v1, v2)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarII")
            print(e)
        return None

    def isSimilarIV(self, img, vec, margin=1.1):
        try:
            v1 = self.embed([img])
            v2 = vec
            dis = distance.cosine(v1, v2)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarIV")
            print(e)
        return None

    def isSimilarVV(self, vec1, vec2, margin=1.0):
        try:
            v1 = vec1
            v2 = vec2
            dis = distance.cosine(v1, v2)
            print(dis)
            if dis > margin:
                return False
            else:
                return True
        except Exception as e:
            print("Error in AIengine.isSimilarVV")
            print(e)
        return None

    @staticmethod
    def prewhiten(x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0/np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    @staticmethod
    def preprocess(images, margin=10, image_size=160, face_extract_algo=face_extract_dnn):
        try:
            faceDetected = True
            aligned_images = []
            detections = []
            for img in images:
                if type(img) is list:
                    img = np.array(img)
                img = to_rgb(img)
                bb = dlib_model.getLargestFaceBoundingBox(img)
                if bb is None:
                    print("dlib couldn't find any face, using dnn...")
                    aligned = img
                    _img, faceDetected = face_extract_dnn(aligned, margin, image_size = image_size)
                    # Comment above lines an uncomment below lines to ignore bad quality face images
                    #continue
                else:
                    aligned = dlib_model.align(img, bb=bb)
                    if aligned is None:
                        print("Error! No aligned photo")
                        aligned = img
                    _img, ss = face_extract_dnn(aligned, margin, image_size = image_size)
                    x, y, w, h = face_utils.rect_to_bb(bb)
                    faceDetected = (x, y, x + w, y + h)
                if faceDetected is False:
                    print("No face detected")
                    print(type(aligned))
                    continue
                detections.append(faceDetected)
                aligned_images.append(_img)
            if len(aligned_images) == 0:
                return images, False
            return np.array(aligned_images), detections
        except Exception as e:
            print("Error in Preprocess ")
            print(e)
            return images, False

    def save(self, modelpath):
        self.modelpath = modelpath
        self.metadata = {'labelEncodeMap': self.labelEncodeMap, 'labelDecodeMap': self.labelDecodeMap,
                         'imagesize': self.image_size, 'similarity_margin': self.margin}
        f = open(modelpath+'/model.meta', 'wb')
        f.write(pickle.dumps(self.metadata))
        f.close()

        f = open(modelpath+'/model.pkl', 'wb')
        f.write(pickle.dumps(self.classifier))
        f.close()
        return True
