from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.conf import settings
import os
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
import pymysql
from sklearn import svm
from django.core.mail import send_mail


global username
global X_train, X_test, y_train, y_test, X, Y
accuracy = []
precision = []
recall = [] 
fscore = []
path = "Dataset"
labels = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    a = (accuracy_score(y_test,predict)*100)
    p = (precision_score(y_test, predict,average='macro') * 100)
    r = (recall_score(y_test, predict,average='macro') * 100)
    f = (f1_score(y_test, predict,average='macro') * 100)
    a = round(a, 3)
    p = round(p, 3)
    r = round(r, 3)
    f = round(f, 3)
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    return algorithm

X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

data = np.load("model/data.npy", allow_pickle=True)
X_train, X_test, y_train, y_test = data

X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))
X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))
y_train1 = np.argmax(y_train, axis=1)
y_test1 = np.argmax(y_test, axis=1) 

svm_cls = svm.SVC()
svm_cls.fit(X_train1, y_train1)
predict = svm_cls.predict(X_test1)
calculateMetrics("SVM", y_test1, predict)

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)    
calculateMetrics("CNN", y_test1, predict)
conf_matrix = confusion_matrix(y_test1, predict)

def Predict(request):
    if request.method == 'GET':
        return render(request, 'Predict.html', {})


# ===== Disease Information (MATCH DATASET LABELS) =====
disease_info = {

    'Healthy_Leaf': {
        'treatment': 'No pesticide required. Maintain balanced fertilization and irrigation.',
        'precautions': 'Regular monitoring and proper farm hygiene.'
    },

    'Black_Sigatoka': {
        'treatment': 'Spray Mancozeb (2g/L) or Propiconazole fungicide every 10–15 days.',
        'precautions': 'Remove infected leaves and maintain good airflow.'
    },

    'Yellow_Sigatoka': {
        'treatment': 'Apply Chlorothalonil or Carbendazim fungicide.',
        'precautions': 'Avoid overhead irrigation and keep field clean.'
    },

    'Bract_Mosaic_Virus': {
        'treatment': 'No chemical cure. Remove infected plants immediately.',
        'precautions': 'Use virus-free planting material and control aphids.'
    },

    'Insect_Pest': {
        'treatment': 'Spray Neem oil or Imidacloprid as recommended.',
        'precautions': 'Regular pest monitoring and biological control.'
    },

    'Moko': {
        'treatment': 'Destroy infected plants and disinfect soil.',
        'precautions': 'Sterilize farm tools and prevent water contamination.'
    },

    'Panama': {
        'treatment': 'Apply Carbendazim soil drenching.',
        'precautions': 'Use resistant varieties and improve drainage.'
    }
}


# ===== Banana Leaf Color Detection =====
def is_banana_leaf(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (200,200))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range (leaf detection)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    green_ratio = np.sum(mask > 0) / (200*200)

    # if enough green pixels → banana leaf
    return green_ratio > 0.15


def PredictAction(request):

    if request.method == 'POST':
        global labels, disease_info

        # ===== Save Uploaded Image =====
        myfile = request.FILES['t1'].read()

        test_path = 'BananaApp/static/test.jpg'

        if os.path.exists(test_path):
            os.remove(test_path)

        with open(test_path, "wb") as file:
            file.write(myfile)

        # ===== Check Banana Leaf First =====
        if not is_banana_leaf(test_path):

            context = {
                'data': "<b>Result :</b> Uploaded image is NOT a Banana Leaf",
                'prediction': "Invalid Image",
                'treatment': "Please upload banana leaf image only.",
                'precautions': "Ensure green banana leaf is clearly visible."
            }
            return render(request, 'UserScreen.html', context)

        # ===== Load Model =====
        cnn_model = load_model("model/cnn_weights.hdf5")

        # ===== Image Preprocessing =====
        image = cv2.imread(test_path)
        img = cv2.resize(image, (32,32))
        img = np.array(img).reshape(1,32,32,3)
        img = img.astype('float32') / 255

        # ===== Prediction =====
        prediction = cnn_model.predict(img)

        confidence = float(np.max(prediction))
        score = round(confidence * 100, 2)
        predict_index = np.argmax(prediction)

        # ===== Confidence Handling =====
        THRESHOLD = 0.55

        if confidence < THRESHOLD:
            display_name = "Low Confidence Prediction"
            treatment = "Upload clearer banana leaf image."
            precautions = "Avoid blur, shadows, or distant capture."

        else:
            disease_name = labels[predict_index]
            display_name = disease_name.replace("_", " ")

            info = disease_info.get(
                disease_name,
                {'treatment':'Not available',
                 'precautions':'Not available'}
            )

            treatment = info['treatment']
            precautions = info['precautions']

        # ===== Display Image =====
        img_display = cv2.resize(image, (700,400))
        img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

        cv2.putText(img_display, display_name, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255,0,0), 2)

        plt.figure(figsize=(8,4))
        plt.imshow(img_display)
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        plt.close()

        # ===== Send Result to HTML =====
        context = {
            'data': f"""
                <b>Predicted As :</b> {display_name}<br/>
                <b>Detected Accuracy :</b> {score}%
            """,
            'img': img_b64,
            'prediction': display_name,
            'treatment': treatment,
            'precautions': precautions
        }

        return render(request, 'UserScreen.html', context)





def TrainModel(request):
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, labels
        global accuracy, precision, recall, fscore, conf_matrix
        output='<table border=1 align=center width=100%><tr><th><font size="3" color="black">Algorithm Name</th><th><font size="3" color="black">Accuracy</th>'
        output += '<th><font size="3" color="black">Precision</th><th><font size="3" color="black">Recall</th><th><font size="3" color="black">FSCORE</th></tr>'
        algorithms = ['SVM Algorithm', 'CNN Algorithm']
        for i in range(len(algorithms)):
            output += '<td><font size="3" color="black">'+algorithms[i]+'</td><td><font size="3" color="black">'+str(accuracy[i])+'</td><td><font size="3" color="black">'+str(precision[i])+'</td>'
            output += '<td><font size="3" color="black">'+str(recall[i])+'</td><td><font size="3" color="black">'+str(fscore[i])+'</td></tr>'
        output+= "</table></br>"
        figure, axis = plt.subplots(nrows=1, ncols=2,figsize=(10, 3))#display original and predicted segmented image
        axis[0].set_title("Confusion Matrix Prediction Graph")
        axis[1].set_title("All Algorithms Comparison Graph")
        ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axis[0]);
        ax.set_ylim([0,len(labels)])    
        df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FSCORE',fscore[0]],
                           ['CNN','Accuracy',accuracy[1]],['CNN','Precision',precision[1]],['CNN','Recall',recall[1]],['CNN','FSCORE',fscore[1]],
                          ],columns=['Parameters','Algorithms','Value'])
        df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar', ax=axis[1])       
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        #plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def LoadDataset(request):    
    if request.method == 'GET':
        global X_train, X_test, y_train, y_test, X, Y, labels
        X_train11, X_test11, y_train11, y_test11 = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
        output = '<font size="3" color="black">Banana Dataset Loaded</font><br/>'
        output += '<font size="3" color="blue">Total images found in Dataset = '+str(X.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">Different Class Labels found in Dataset = '+str(labels)+'</font><br/><br/>'
        output += '<font size="3" color="black">Dataset Train & Test Split details</font><br/>'
        output += '<font size="3" color="blue">80% dataset images used to train all Architectures = '+str(X_train11.shape[0])+'</font><br/>'
        output += '<font size="3" color="blue">20% dataset images used to test all Architectures = '+str(X_test11.shape[0])+'</font><br/>'
        context= {'data':output}
        return render(request, 'UserScreen.html', context)

def RegisterAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'banana',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break                
        if output == "none":
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'banana',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO register(username,password,contact,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = "Signup process completed. Login to perform Banana Leaf Detection"
        context= {'data':output}
        return render(request, 'Register.html', context)    

def UserLoginAction(request):
    global username
    if request.method == 'POST':
        global username
        status = "none"
        users = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'banana',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM register")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == users and row[1] == password:
                    username = users
                    status = "success"
                    break
        if status == 'success':
            context= {'data':'Welcome '+username}
            return render(request, "UserScreen.html", context)
        else:
            context= {'data':'Invalid username'}
            return render(request, 'UserLogin.html', context)

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})



def ForgotPassword(request):
    if request.method == 'GET':
        return render(request, 'ForgotPassword.html', {})


def ForgotPasswordAction(request):

    if request.method == 'POST':
        email = request.POST.get('email', False)

        con = pymysql.connect(host='127.0.0.1',
                              port=3306,
                              user='root',
                              password='root',
                              database='banana',
                              charset='utf8')

        with con:
            cur = con.cursor()
            cur.execute("SELECT username,password FROM register WHERE email=%s",(email,))
            row = cur.fetchone()

        if row:
            username = row[0]
            password = row[1]

            subject = "Banana Leaf Detection - Password Recovery"
            message = f"""
Hello {username},

Your account password is:

Password : {password}

Login again using this password.
"""

            send_mail(
                subject,
                message,
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False,
            )

            context={'data':'Password sent to your registered email'}
            return render(request,'UserLogin.html',context)

        else:
            context={'data':'Email not registered'}
            return render(request,'ForgotPassword.html',context)
