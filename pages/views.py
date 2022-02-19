from django.shortcuts import render,redirect
from.models import Diabetes
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create your views here.
is_first_time = 0
glob_data={}

def home(request):
    return render(request,"pages/home.html")

@csrf_exempt
@login_required
def predict(request):
    if request.method == "GET":
        context={}
        return render(request,"pages/predict.html",context) 

    elif request.method == "POST":
        is_first_time = 0

        data = pd.read_csv('static/diabetes.csv')
        X = data.drop("Outcome",axis = 1)
        Y = data['Outcome']

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
        model = LogisticRegression()
        model.fit(X_train,Y_train)

        Pregnancies = float(request.POST['Pregnancies'])
        Glucose = float(request.POST['Glucose'])
        Blood_Pressure = float(request.POST['Blood_Pressure'])
        Skin_Thickness = float(request.POST['Skin_Thickness'])
        Insulin = float(request.POST['Insulin'])
        Bmi_val = float(request.POST['Bmi_val'])
        Diabetes_Pedigree_Function = float(request.POST['Diabetes_Pedigree_Function'])
        Age = float(request.POST['Age'])

        data_arr = [Pregnancies,Glucose,Blood_Pressure,Skin_Thickness,Insulin,Bmi_val,Diabetes_Pedigree_Function,Age]
        prediction = model.predict([data_arr])

        if prediction == [1]:
            result = "Postive"
        else:
            result = "Negative"

        all_data = {
            "Pregnancies":Pregnancies,
            "Glucose":Glucose,
            "Blood_Pressure":Blood_Pressure,
            "Skin_Thickness":Skin_Thickness,
            "Insulin":Insulin,
            "Bmi_val":Bmi_val,
            "Diabetes_Pedigree_Function":Diabetes_Pedigree_Function,
            "Age":Age,
            "result":result,
            "messages":[
                {"tags":"warning","text":"Result Fetched Successfully"}
            ]
        }
        global glob_data
        glob_data = all_data.copy()

        b = Diabetes.objects.create(
            user=request.user,
            Pregnancies=Pregnancies,
            Glucose=Glucose,
            Blood_Pressure=Blood_Pressure,
            Skin_Thickness=Skin_Thickness,
            Insulin=Insulin,
            Bmi_val=Bmi_val,
            Diabetes_Pedigree_Function=Diabetes_Pedigree_Function,
            Age=Age,
            Result=result
        )
        b.save()

        # user = User.objects.create_user(username="name", email="email@mail.com", password="Pass12345")
        # post_1 = Post(name="testname", email="testemail", gender="Monsieur", number="23233", author=user)

        # Printing Data
        # print(glob_data)
        # print("1st")

    return redirect("result")

def result(request):

    # context={}
    # is_first_time = 0
    # if is_first_time == 0:
    #     content = all_data
    #     is_first_time = 1
    # else:
    #     pass

    # Printing Data
    # print(glob_data)
    # print("2st")
    global glob_data
    return render(request,"pages/result.html",glob_data) 

# def recent_results():
