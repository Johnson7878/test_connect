from flask import Flask, request, jsonify
from fastai.tabular import *
from fastai.tabular.all import *
import os
from mysql.connector import Error
import mysql.connector

app = Flask(__name__)


@app.route('/getinps/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    syear = request.args.get("syear", None)
    eyear = request.args.get("eyear", None)
    week = request.args.get("week", None)
    home_team = request.args.get("home_team", None)
    away_team = request.args.get("away_team", None)
    des_year = request.args.get("des_year", None)
    key = request.args.get("key", None)

    # For debugging
    print(f"Received: {syear}")
    print(f"Received: {eyear}")
    print(f"Received: {week}")
    print(f"Received: {home_team}")
    print(f"Received: {away_team}")
    print(f"Received: {des_year}")
    print(f"Received: {key}")

    response = {}

    # Check if the user sent a name at all
    if not syear:
        response["ERROR"] = "No year found. Please send a year."
    elif not eyear:
        response["ERROR"] = "No year found. Please send a year."
    elif not week:
        response["ERROR"] = "No week found. Please send a week."
    elif not home_team:
        response["ERROR"] = "No team found. Please send a team."
    elif not away_team:
        response["ERROR"] = "No team found. Please send a team."
    elif not des_year:
        response["ERROR"] = "No year found. Please send a year."
    elif key != "28c53a5c-f930-4069-92a9-c1999a17c66b":
        return jsonify("404")
    # Check if the user entered a number
    elif not (str(syear).isdigit() and str(eyear).isdigit() and str(week).isdigit() and str(des_year).isdigit()):
        response["ERROR"] = "WE MUST HAVE A NUMBER."
    else:
        response["MESSAGE"] = f"Inputs are {syear} , {eyear} , {week} , {home_team} , {away_team}, {des_year}"

    # Return the response in json format
    

    connection = mysql.connector.connect(
    host="us-east.connect.psdb.cloud",
    database="gatornetics",
    user="e24p22mu6wajaaxgbjx9",
    password="pscale_pw_V89FpSn6pbE9ildQotJoKnajhsoOVTGXF2PwYXUV1V1",
    ssl_mode = "VERIFY_IDENTITY",
    ssl      = {
        "ca": "/etc/ssl/certs/ca-certificates.crt"
    }
    )


    cursor = connection.cursor()
    sql = "SELECT * FROM data WHERE year = " + syear
    cursor.execute(sql)
    from pandas import DataFrame
    df = DataFrame(cursor.fetchall())
    df.columns = ["id", "year" , "week" , "neutral_site", "home_team", "home_conference", "home_points", "home_elo", "away_team", "away_conference", "away_points", "away_elo", "spread", "margin"]
    connection.close()


    learn = load_learner('talking_tech_neural_net')
    dumpster = df.query(f"year == {des_year}")
    pdf = dumpster.copy()
    dl = learn.dls.test_dl(pdf)
    pdf['predicted'] = learn.get_preds(dl=dl)[0].numpy()
    temp = pdf.loc[(pdf['home_team'] == home_team) & (pdf['away_team'] == away_team) & (pdf['week'] == int(week))].values.flatten().tolist()
    answer = [temp[1], temp[2], temp[4], temp[8], temp[6], temp[10], temp[12], temp[14]]
    return jsonify(answer)


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Enter in parameters to the ML model in the url above!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
