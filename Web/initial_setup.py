import pymongo
from werkzeug.security import generate_password_hash, check_password_hash

db = pymongo.MongoClient("mongodb://localhost:27017/")
df = db['sketchula']  # Create Database

dusers = df['users']     # Create Collections

da = {"_id": "admin", "password": generate_password_hash("admin"),"name":"Ashish Kumar Singh", "type": "admin", 
    "email": "admin@ardulous.io", "keys":["default", "testing"], "calls":list([])}
dusers.save(da)

dapis = df['api_keys']

da = {"_id":"default", "model_type":"vanilla", "model_id":"vanilla", "user":"admin"}
dapis.save(da)
da = {"_id":"testing", "model_type":"private", "model_id":"second", "user":"admin"}
dapis.save(da)


