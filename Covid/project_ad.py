# -*- coding: utf-8 -*-
"""
Created on Wed May 27 12:04:27 2020
@author: Team Phi - 6
Project Area Denial
"""
#import libraries
from joblib import dump, load
import pandas as pd
import numpy as np

# Flask
from flask import Flask, redirect, url_for, render_template, request
app = Flask(__name__)

@app.route("/home")
@app.route("/")
def home():
    """Home Page"""
    return render_template("index.html")

@app.route("/members")
def members():
    """Members Page"""
    return render_template("members.html")

@app.route("/about")
def about():
    """About Page"""
    return render_template("about.html")

@app.route("/reference")
def reference():
    """Reference Page"""
    return render_template("reference.html")


@app.route("/data")
def data():
    """Answer Page"""
    return render_template('data.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("I was here 1")
    if request.method == 'POST':
        try:
            city = request.form['city']
            temperature = request.form['temperature']
            pm10 = request.form['pm10']
            humidity = request.form['humidity']
            no2 = request.form['no2']
            # check city and input stored pop and sqkm
            if city == 'CALOOCAN CITY':
                population = 1500000
                sqkm = 53.334

            elif city == 'CITY OF LAS PIÑAS':
                population = 590000
                sqkm = 32.69

            elif city == 'CITY OF MAKATI':
                population = 510383
                sqkm = 21.57

            elif city == 'CITY OF MALABON':
                population = 365525
                sqkm = 19.76

            elif city == 'CITY OF MANDALUYONG':
                population = 300000
                sqkm = 11.06

            elif city == 'CITY OF MANILA':
                population = 1600000
                sqkm = 42.88

            elif city == 'CITY OF MARIKINA':
                population = 450741
                sqkm = 21.52

            elif city == 'CITY OF MUNTINLUPA':
                population = 504509
                sqkm = 46.7

            elif city == 'CITY OF NAVOTAS':
                population = 249463
                sqkm = 10.77

            elif city == 'CITY OF PARAÑAQUE':
                population = 665822
                sqkm = 46.47

            elif city == 'CITY OF PASIG':
                population = 755300
                sqkm = 48.45

            elif city == 'CITY OF SAN JUAN"':
                population = 122180
                sqkm = 7.77

            elif city == 'CITY OF VALENZUELA':
                population = 620422
                sqkm = 47.02

            elif city == 'PASAY CITY':
                population = 416522
                sqkm = 18.5

            elif city == 'PATEROS':
                population = 416522
                sqkm = 18.5

            elif city == 'QUEZON CITY':
                population = 2761720
                sqkm = 166.2

            else:
                population = 644473
                sqkm = 47.88

            dict_client = {}
            dict_client["City"] = city
            dict_client[" Population "] = population
            dict_client["Square km"] = sqkm
            dict_client["temperature"] = temperature
            dict_client["pm10"] = pm10
            dict_client["humidity"] = humidity
            dict_client["no2"] = no2

            df_client = pd.DataFrame.from_dict(dict_client, orient='index').T
            print(df_client)

            print("#numeric scaler")
            numerics = [' Population ', 'Square km', 'temperature', 'pm10', 'humidity', 'no2']
            scaler = load('num_scaler.joblib')
            print("loaded scaler")
            scaled = scaler.transform(df_client[numerics])  # import the pkl file for the scaler and ohe
            pr_num = pd.DataFrame(scaled, columns=numerics)
            print(scaled)

            print('#cat encoder')
            column_trans = load('cat_scaler.joblib')
            cat = column_trans.transform(df_client)
            cat_encoder = column_trans.named_transformers_["cat"]
            x = cat_encoder.categories_
            new_list = []

            for i in x:
                for y in i[1:]:
                    new_list.append(y)

            new_cat = pd.DataFrame(cat, columns=new_list)

            test = pd.concat([pr_num, new_cat], axis=1)  # transformed and scaled values
            print(test)
            # load model
            model = load('car_predict.joblib')
            pred = model.predict(test)  # calculated value
            pred = (round(int(pred), 2))
            print(pred)


        except ValueError as e:
            print(e)
            return 'Please check the values! - Team Phi-6'

    return render_template('predict.html', prediction=pred, city=city)




@app.route("/no_page")
def no_page():
    """Redirect page"""
    return "<h1>The website you are looking for is not found.<h1>"
    
@app.route("/<name>")
def user(name):
    """This is for <name> or mispelled webpages"""
    return redirect(url_for("no_page"))


#Flask
if __name__ == "__main__":
        app.run(host='0.0.0.0')
#        app.run(debug=True)