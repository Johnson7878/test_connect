from flask import Flask, request, jsonify
import pickle
import random
from flask_cors import CORS
#from multiprocessing import Pool
from joblib import parallel_backend
#from ray.util.joblib import register_ray
#import ray
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os


app = Flask(__name__)
CORS(app)


# def calc(offenseScore, defenseScore, ytg, offVal, defVal, classifier):
#     X = np.array([offenseScore, defenseScore, ytg, offVal, defVal], dtype=object)
#     y_pred = classifier.predict(X.reshape(1, -1))
#     return y_pred

#def calc1(X1, offenseScore, defenseScore, ytg, offVal, defVal):
    #return(predictedOutcome)

@app.route('/getinps/', methods=['GET'])

def respond():
    # Retrieve the name from the url parameter /getmsg/?name=
    
    defenseScore = int(request.args.get("defenseScore", None))
    offenseScore = int(request.args.get("offenseScore", None))
    ytg = int(request.args.get("ytg", None))
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

    

    #add in URL params for ytg, offenseScore, defenseScpre

    # defenseScore = int(defenseScore)
    # offenseScore = int(offenseScore)
    # ytg = int(ytg)

    
    #/////////////////////////////////////////////////////////////////////////
    start = time.perf_counter()
    f = open('coleClassifier.pkl', 'rb')
    classifier = pickle.load(f)
    f.close()
    teams = pd.read_csv('teamsRatings.csv')
    teams.reset_index(drop=True)
    mergedData = pd.read_csv('coleBigData.csv')
    #print("\nOffense: ", offenseScore ,"          ",defenseScore," :Defense")
    #print("\n          Time: 0:01")
    #print("\nYards to go: ", ytg)
    offVal = teams['offPPA'].loc[teams['school'] == offense_team]
    defVal = teams['defPPA'].loc[teams['school'] == defense_team]
    #/////////////////////////////////////////////////////////////////////////




    #multiprocessing
    # with Pool() as p:
    #     y_pred = p.starmap(calc, [(offenseScore, defenseScore, ytg, offVal, defVal, classifier),])
    

    #Ray
    #ray.init()
    X = np.array([offenseScore, defenseScore, ytg, offVal, defVal], dtype=object)
    #register_ray()
    with parallel_backend('threading', n_jobs=1):
        y_pred = classifier.predict(X.reshape(1, -1))
    #ray.shutdown()

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
    
    #with Pool() as p1:
    #predictedOutcome = p1.starmap(calc1, [(X1, offenseScore, defenseScore, ytg, offVal, defVal),])
    #predictedOutcome = calc1(X1, offenseScore, defenseScore, ytg, offVal, defVal)
    with parallel_backend('threading', n_jobs=1):
        kmeans = KMeans(n_clusters=19, max_iter=500, algorithm = 'auto')
        kmeans.fit(X1)
        test = np.array([ytg, offenseScore, defenseScore, offVal, defVal], dtype=object)
        predictedOutcome = kmeans.predict(test.reshape(1, -1))
    
    # zaza = 0
    # for temp in predictedOutcome[0]:
    #     zaza = temp
    # predictedOutcome = zaza
    if(play == "Field Goal"):
        predictedOutcome = 0
        offenseScore += 3
    if(predictedOutcome >= ytg):
        offenseScore += 7
    if(y_pred == 0):
        predictedOutcome *= -1
    
    
    
    #/////////////////////////////////////////////////////////////////////////
    #print("Offense: ", offenseScore ,"          ",defenseScore," :Defense")
    #print("\n          Time: 0:00")
    #print("\nYards to go: ", ytg)
    #print("\nYards gained: ", predictedOutcome)
    end =  time.perf_counter()
    print("\nElapsed = {}s".format((end - start)))
    #/////////////////////////////////////////////////////////////////////////

    

    answer = (int(predictedOutcome))
    return jsonify(answer)


@app.route('/')
def index():
    # A welcome message to test our server
    return "<h1>Enter in parameters to the ML model in the url above!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
