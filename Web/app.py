import string 
from flask import * 
from Database import *
from flask_sessionstore import Session
import json
from werkzeug.utils import secure_filename
import subprocess
from bson import ObjectId
import hashlib
import time 
import requests

from Inference import *

app = Flask(__name__)
app.config.update(
    DATABASE = 'Sketchula'
)
SESSION_TYPE = "filesystem"
app.config.from_object(__name__)
#Session(app)
AIengine = InferenceWrapper(Default_InferenceEngine())

global db  
db = Database("mongodb://localhost:27017/")

# Set the secret key to some random bytes. Keep this really secret!
import os 
import random
app.secret_key = os.urandom(32)#bytes(str(hex(random.getrandbits(128))), 'ascii')

from flask_mail import Mail, Message


app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'tagclub.vitu@gmail.com'#'tagclub.vituniversity@gmail.com'
app.config['MAIL_PASSWORD'] = 'TAGnumba1'#'tagTHEclub@19'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

@app.route("/send_email")
def send_email(email, uid, otp):
    try:
        msg = Message('Hello ' + uid, sender=app.config.get("MAIL_USERNAME"), recipients = [email], body = "Hello! " + uid + ", Greetings from Tag club. Please use the following otp : " + otp + " on the link http://phoenix.tagclub.in/registerVerify" )
        mail.send(msg)
        return True
    except Exception as e:
        print(e)
        return None

@app.errorhandler(404)
def page_not_found(e):
    return render_template("/404.html")

@app.route("/", methods=["GET", "POST"])        # Home Page
@app.route("/home", methods=["GET", "POST"])    # Future Home Page
def home():
    if "login" in session:
        return redirect("/dashboard")
    else:
        return render_template('/index.html')

@app.route("/login", methods=["GET", "POST"])
@app.route("/login_user", methods=["GET", "POST"])
def login_user():
    if "login" in session:
        return redirect("/dashboard")
    elif request.method == "POST":
        try:
            uid = request.form['uname']
            upass = request.form['pass']
            if(uid == '' or upass == ''):
                return render_template('/login.html')
            val = db.validateUser(uid, upass)
            if val:
                if val == "unverified":     # Don't let them login!
                    return redirect("/registerVerify")
                session["login"] = uid
                session["feedpos"] = 0
                session["type"] = val
                #session["database"] = Database("http://admin:ashish@localhost:5984")
                if val == "admin":
                    return redirect("/admin")
                else:
                    return redirect("/dashboard")
            else:
                return "Incorrect Username/Password"
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    return render_template('/login.html')

@app.route("/register", methods=["GET", "POST"])
@app.route("/register_user", methods=["GET", "POST"])
def register_user():
    if "login" in session:
        return redirect("/dashboard")
    elif request.method == "POST":
        global db
        try:
            data = dict(request.form)
            status = db.userExists(data)
            if status == 2:
                return render_template('/signup.html', resp = "alert('Username Already Taken!');")
            elif status == 3:
                return render_template('/signup.html', resp = "alert('Email Already Taken!');")
            #if data['pass'] != data['cpass']:
            #    return render_template('/register.html', resp = "alert('Passwords do not match!');")
                
            data['otp'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
            
            if send_email(data['email'], data['uname'], data['otp']) is None:
                return jsonify("Problem in sending Email!")
                
            if db.createUser(data) == 1:
               return render_template("/registerVerify.html", uid = data['uname'], resp = '')
            else: return render_template('/signup.html', resp = "alert('Username Already Taken');")
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    else:
        return render_template('/signup.html', resp = "")


@app.route("/registerVerify", methods=["GET", "POST"])
def registerVerify():
    if "login" in session:
        return redirect("/dashboard")
    elif request.method == "POST":
        global db
        try:
            uid = request.form['uname']
            otp = request.form['otp']

            if db.verifyOtpUser(uid, otp):
                # Create default API keys for the user
                val1 = db.createNewAPIKey(uid, uid + "_vanilla", "vanilla", "vanilla")
                val2 = db.createNewAPIKey(uid, uid + "_global", "global", "global")
                #if val1 and val2:
                return render_template("/success.html")
            else: return render_template("/registerVerify.html", uid = uid, resp = "alert('Wrong otp!');")
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    else:
        return render_template("/registerVerify.html", uid = "", resp = "")
        #return render_template('/landing/login/register.html', resp = "")


@app.route("/resendOTP", methods=["GET", "POST"])
def resendOTP():
    if "login" in session:
        return redirect("/dashboard")
    elif request.method == "POST":
        global db
        try:
            email = request.form['email']
            otp,uname = db.getOTPbyEmail(email)
            g = send_email(email, uname, otp)
            if g:
               return render_template("/resendOTP.html", resp = "alert('OTP sent again, may take few minutes! Contact us on Facebook if it dosent work')");
            else: return render_template("/resendOTP.html", resp = "alert('Email address not found, Incorrect or Not Registered?')");
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    else:
        return render_template("/resendOTP.html", resp = "")
        #return render_template('/landing/login/register.html', resp = "")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    global db
    del db 
    db = Database("mongodb://localhost:27017/")
    session.pop('login', None)
    session.pop('feedpos', None)
    return redirect("/login_user")#render_template("/login_user.html")


########################################## ADMIN Dashboard and Secret Stuffs ##########################################

@app.route("/admin", methods=["GET", "POST"])
def admin():
    if ("login" in session) and ("admin" in session["type"]):
        print("HLP")
        try:
            return redirect("/dashboard")#render_template('admin.html')
        except Exception as ex:
            print(ex)
            return render_template("/500.html")
    return render_template('/404.html')


############################################ Dashboard and internal stuffs ############################################


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "login" in session: 
        if request.method == "POST": # Redeem Flags
            #pp = request.form[]
            return render_template('/profile.html')
        ss = session['login'] 
        info = db.getUserInfo(ss)
        try:
            apiKeys = db.getUserApiKeys(ss)
        except Exception as e:
            print(e)
        print(info)
        return render_template('/profile.html')#, acc_level = info['type'], api_call_count = len(info['calls']), 
                                #api_keys = json.dumps({"data":info['keys']}), api_key_count = len(info['keys']), profile_id = info['_id'], 
                                #profile_name = info['name'], profile_email = info['email'])
    else:
        return redirect("/login_user")
    return render_template('/500.html')

############################################ RESTFUL APIs, STATEFUL ############################################

# The Restful APIs are defined here
import numpy as np
import cv2
from io import BytesIO, StringIO
import base64
import re
import json
from PIL import Image
import pickle
from skimage.io import imsave
import os

global contentImg

@app.route("/api/generate", methods=['POST'])
def api_generate():
    if "login" not in session: 
        return redirect("/login_user")
    try:
        print("non json request")
        image = request.form['img']
        model_name = request.form['modelname']
        model_type = request.form['modeltype']
    except:
        data = request.get_json(force=True)
        print("json request")
        image = request.values['img']
        model_name = request.values['modelname']
        model_type = request.values['modeltype']
        #print(data)
    try:
        image_data = re.sub('^data:image/.+;base64,', '', image)
        rawimage = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        image = pickle.dumps(rawimage).hex()
        #print(image)
        session['image'] = image
        global contentImg
        contentImg = image
        if(model_name != 'identity'):
            # Just save it locally
            user = session['login'] 
            result = AIengine.generate({"data":image, "type":model_type, "modelname":model_name, "user":user})
            print("Got results")
            rawresult = pickle.loads(bytes.fromhex(result))
            print(type(result))
            print(rawresult.shape)
            callID = db.logApiCall({"timestamp":time.time(), "user":user, "type":model_type, "modelname":model_name, "data":image, "result":result})
            im = Image.fromarray(rawresult)
        else:
            im = rawimage 
        resultUrl = "static/generated/" + user + "/" + str(callID) + ".png"
        try:
            im.save(resultUrl)
        except:
            os.system("mkdir " + "static/generated/" + user)
            im.save(resultUrl)
        #return send_file(strIO, mimetype='image/png')
        print("Got till here")
        return jsonify({"url":resultUrl})#jsonify({"callid":callID, "result":result['text']})
    except Exception as e:
        print(e)
        return e
    
@app.route("/api/transfer", methods=['POST'])
def api_styleTransfer():
    if "login" not in session: 
        return redirect("/login_user")
    try:
        print("non json request")
        style = request.values['style']
        model_name = request.values['modelname']
        model_type = request.values['modeltype']
    except:
        data = request.get_json(force=True)
        print("json request")
        style = request.values['style']
        model_name = request.values['modelname']
        model_type = request.values['modeltype']
        #print(data)
    try:
        #print(style)
        global contentImg
        image = contentImg
        return jsonify({"url":"static/img/test5.png"})
        #image = session['image']
        
        
        image_data = re.sub('^data:image/.+;base64,', '', style)
        rawimage = Image.open(BytesIO(base64.b64decode(image_data))).convert('RGB')
        style = pickle.dumps(rawimage).hex()

        user = session['login'] 
        result = AIengine.transfer({"data":image, "style":style, "type":model_type, "modelname":model_name, "user":user})
        print("Got results")
        rawresult = pickle.loads(bytes.fromhex(result))
        print(type(result))
        callID = db.logApiCall({"timestamp":time.time(), "user":user, "type":model_type, "modelname":model_name, "data":image, "style":style, "result":result})
        im = rawresult#Image.fromarray(rawresult)
        resultUrl = "static/generated/" + user + "/" + str(callID) + ".png"
        try:
            im.save(resultUrl)
        except:
            os.system("mkdir " + "static/generated/" + user)
            im.save(resultUrl)
        #return send_file(strIO, mimetype='image/png')

        return jsonify({"url":resultUrl})#jsonify({"callid":callID, "result":result['text']})
    except Exception as e:
        print(e)
        return e
