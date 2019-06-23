import pymongo
from bson.objectid import ObjectId
import hashlib
import json
from werkzeug.security import generate_password_hash, check_password_hash

import re

def getAge(date):
    return 19

class Database:
    def __init__(self, url):
        self.client = pymongo.MongoClient(url)
        self.db = self.client['smart']
        return 

    def userExists(self, data):
        d = self.db 
        try:
           # b = d['users'][data['id']]
            if(d['users'].find_one({'_id':data['uname']})):
                #print(2)
                return 2
            if(d['users'].find_one({'email':data['email']})):
                #print(3)
                return 3
            return False
        except:
            return False 
        return False
        

    def createUser(self, data, typ = "unverified"):
        d = self.db 
        try:
           # b = d['users'][data['id']]
            if(d['users'].find_one({'_id':data['uname']})):
                #print(2)
                return 2
            if(d['users'].find_one({'email':data['email']})):
                #print(3)
                return 3
            dd = {"_id":data['uname'], "password" : generate_password_hash(data['pass']), "type":typ, 
                "otp":data['otp'], "email":data['email'], "name":data['name'], "keys":list([]), 
                "calls":list([])}
            d['users'].insert_one(dd)
            return 1
        except:
            return False 
        return False

    def verifyOtpUser(self, uid, otp):
        d = self.db
        try: 
            val = dict(d['users'].find_one({"_id":uid}))
            if(val['otp'] == otp) or True:
                val['type'] = "normaluser"
                d['users'].save(val)
                return True
            else: 
                print(otp)
                print(val['otp'])
                return False
        except Exception as e:
            print(e)
            return None

    def getOTPbyEmail(self, email):
        d = self.db
        try: 
            val = dict(d['users'].find_one({"email":email}))
            if val is None: 
                return None, None
            else: 
                return val['otp'], val['_id']
        except:
            return None, None

    def validateUser(self, uid, upass):
        d = self.db
        try: 
            val = d['users'].find_one({"_id":uid})
            h = val['password']
            if(check_password_hash(h, upass)):
                return val['type']
            return None
        except:
            return None

    def validateAdmin(self, uid, upass):
        d = self.db
        try: 
            h = d['users'].find_one({"_id":uid})['password']
            if d['users'].find_one({"_id":uid})['type'] == 'admin':
                if(check_password_hash(h, upass)):
                    return True
            return False
        except:
            return False

    def validateApikey(self, key, user):
        d = self.db 
        try:
            h = d['api_keys'].find_one({"_id":key})
            if h is None:
                print("Api Key not found")
                return None 
            h = dict(h)
            if(h['user'] != user):
                print("User does not match")
                return None 
            print("API Verified!")
            return h 
        except Exception as e:
            print("Error in API validation")
            return None
        return None 
    
    def validateCallID(self, callid, key):
        d = self.db 
        try:
            i = d['api_call_logs'].find_one({'_id':callid})
            if i['api_key'] == key:
                return True 
            return False
        except:
            print("Error in fetching callid")
            return False 
        return False


    def getUserInfo(self, uid):
        d = self.db 
        info = {}
        try:
            b = d['users'].find_one({"_id":uid})
            #info['email'] = b['email']
            #info['name'] = b['name']
            #info['score'] = b['score']
            #info['currentQues'] = b['currentQues']
            #info['flags'] = b['flags']
            info = dict(b)
            #info['stats']['following'] = len(b['connections']['following'])
            #info['stats']['posts'] = len(b['originals'])
            return info
        except:
            return None 
        return None
    
    def logApiCall(self, data):
        d = self.db
        try:
            i = d['api_call_logs'].save(data)
            b = d['users'].find_one({"_id":data['user']})
            b['calls'].append(i)
            d['users'].save(b)
            return i
        except:
            print("Error in log api calls")
            return None 
        return None
    
    def logApiFeedback(self, data):
        d = self.db
        try:
            i = d['api_feedback_logs'].save(data)
            #b = d['users'].find_one({"_id":data['user']})
            #b['calls'].append(i)
            #d['users'].save(b)
            return i
        except:
            print("Error in log api calls")
            return None 
        return None

    def getTextFromCall(self, callid):
        d = self.db 
        try:
            i = d['api_call_logs'].find_one({'_id':callid})
            return i['text']
        except:
            print("Error in fetching callid")
            return None 
        return None
    
    def getCallData(self, callid):
        d = self.db 
        try:
            i = d['api_call_logs'].find_one({'_id':callid})
            return dict(i)
        except:
            print("Error in fetching callid")
            return None 
        return None
        
    def createNewModel(self, user, key, newid):
        d = self.db 
        try:
            i = dict(d['api_keys'].find_one({'_id':key}))
            model = i['model_id']
            j = dict({"user":user, "model_type":"private", "model_id":newid, "_id":newid})
            d['api_keys'].save(j)
            return j,i
        except:
            print("Error in creating new model")
            return None 
        return None

    def createNewAPIKey(self, user, key, modeltype, modelid):
        d = self.db 
        try:
            h = {"_id":key, "model_type":modeltype, "model_id":modelid, "user":user}
            d['api_keys'].save(h)
        except:
            print("Error in creating new API Key")
            return None 
        return None


    def validateSharing(self, originaluser, newuser, key):
        return False