from flask import Flask, request, jsonify
import pickle
import random
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os


app = Flask(__name__)


@app.route('/getinps/', methods=['GET'])
def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    
    defenseScore = request.args.get("defenseScore", None)
    offenseScore = request.args.get("offenseScore", None)
    ytg = request.args.get("ytg", None)
    offense_team = request.args.get("offense_team", None)
    defense_team = request.args.get("defense_team", None)
    play = request.args.get("play", None)
    key = request.args.get("key", None)

    # For debugging
    print(f"Received: {defenseScore}")
    print(f"Received: {offenseScore}")
    print(f"Received: {ytg}")
    print(f"Received: {offense_team}")
    print(f"Received: {defense_team}")
    print(f"Received: {play}")
    print(f"Received: {key}")

    response = {}

    # Check if the user sent a name at all
    if not defenseScore:
        response["ERROR"] = "No Defense Score found. Please send a Score."
    elif not offenseScore:
        response["ERROR"] = "No Offense Score found. Please send a Score."
    elif not ytg:
        response["ERROR"] = "No ytg found. Please send a ytg."
    elif not offense_team:
        response["ERROR"] = "No team found. Please send a team."
    elif not defense_team:
        response["ERROR"] = "No team found. Please send a team."
    elif not play:
        response["ERROR"] = "Play not found. Please send a play."
    elif key != "28c53a5c-f930-4069-92a9-c1999a17c66b":
        return jsonify("404")
    

    # Return the response in json format
    
    # load_dotenv()

    # connection = mysql.connector.connect(
    # host=os.getenv("HOST"),
    # database=os.getenv("DATABASE"),
    # user=os.getenv("IDENTITY"),
    # password=os.getenv("PASSWORD"),
    # ssl_ca=os.getenv("SSL_CERT")
    # )


    # cursor = connection.cursor()
    # sql = "SELECT * FROM data WHERE year = " + syear
    # cursor.execute(sql)
    # from pandas import DataFrame
    # df = DataFrame(cursor.fetchall())
    # df.columns = ["id", "year" , "week" , "neutral_site", "home_team", "home_conference", "home_points", "home_elo", "away_team", "away_conference", "away_points", "away_elo", "spread", "margin"]
    # connection.close()


    #add in URL params for ytg, offenseScore, defenseScpre

    defenseScore = int(defenseScore)
    offenseScore = int(offenseScore)
    ytg = int(ytg)

    f = open('coleClassifier.pkl', 'rb')
    classifier = pickle.load(f)
    f.close()
    teams = pd.read_csv('teamsRatings.csv')
    teams.reset_index(drop=True)
    mergedData = pd.read_csv('coleBigData.csv')

    #defenseScore = random.randint(10,60)
    #offenseScore = defenseScore - random.randint(4,7)
    #ytg = random.randint(1,20)
    print("\nOffense: ", offenseScore ,"          ",defenseScore," :Defense")
    print("\n          Time: 0:01")
    print("\nYards to go: ", ytg)

    offVal = teams['offPPA'].loc[teams['school'] == offense_team]
    defVal = teams['defPPA'].loc[teams['school'] == defense_team]

    X = np.array([offenseScore, defenseScore, ytg, offVal, defVal])
    y_pred = classifier.predict(X.reshape(1,-1))
    print(y_pred)

    if(y_pred == 1):
        regData = mergedData[mergedData['playOutcomeClass']==1]
    else:
        regData = mergedData[mergedData['playOutcomeClass']==0]

    if(play == "Pass"):
        regData = regData[regData['play_type'] != "Rush"]
        regData = regData[regData['play_type'] != "Rushing Touchdown"]
    else:
        regData = regData[regData['play_type'] != "Pass Reception"]
        regData = regData[regData['play_type'] != "Passing Touchdown"]

    X1 = regData[['ytg','offense_score','defense_score','offVal', 'defVal']]
    y1 = regData['yg']
    kmeans = KMeans(n_clusters=19, max_iter=500, algorithm = 'auto')
    kmeans.fit(X1)

    test = np.array([ytg, offenseScore, defenseScore, offVal, defVal])
    predictedOutcome = kmeans.predict(test.reshape(1, -1))
    if(play == "Field Goal"):
        predictedOutcome = 0
        offenseScore += 3
    if(predictedOutcome >= ytg):
        offenseScore += 7
    if(y_pred == 0):
        predictedOutcome *= -1

    print("Offense: ", offenseScore ,"          ",defenseScore," :Defense")
    print("\n          Time: 0:00")
    print("\nYards to go: ", ytg)
    print("\nYards gained: ", predictedOutcome)

    answer = (int(predictedOutcome))
    return jsonify(answer)


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Enter in parameters to the ML model in the url above!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
