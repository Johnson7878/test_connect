from flask import Flask, request, jsonify
import cfbd
import numpy as np
import pandas as pd

from fastai.tabular import *
from fastai.tabular.all import *
#configure cfbd client-host connection
configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = 'uHZyMKEnExs1jxdAWvwmblkR3+vRvnTFT7Ene2/kAMsZefXA3tabMMugpG6hQWh4'
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_config = cfbd.ApiClient(configuration)
teams_api = cfbd.TeamsApi(api_config)
ratings_api = cfbd.RatingsApi(api_config)
games_api = cfbd.GamesApi(api_config)
stats_api = cfbd.StatsApi(api_config)
betting_api = cfbd.BettingApi(api_config)
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
    
    games = []
    lines = []

    for year in range(int(syear), int(eyear)):
        response = games_api.get_games(year=year)
        games = [*games, *response]

        response = betting_api.get_lines(year=year)
        lines = [*lines, *response]
    games = [g for g in games if g.home_conference is not None and g.away_conference is not None and g.home_points is not None and g.away_points is not None]
    games = [
    dict(
        id = g.id,
        year = g.season,
        week = g.week,
        neutral_site = g.neutral_site,
        home_team = g.home_team,
        home_conference = g.home_conference,
        home_points = g.home_points,
        home_elo = g.home_pregame_elo,
        away_team = g.away_team,
        away_conference = g.away_conference,
        away_points = g.away_points,
        away_elo = g.away_pregame_elo
    ) for g in games]
    for game in games:
        game_lines = [l for l in lines if l.id == game['id']]

        if len(game_lines) > 0:
            game_line = [l for l in game_lines[0].lines if l.provider == 'consensus']

            if len(game_line) > 0 and game_line[0].spread is not None:
                game['spread'] = float(game_line[0].spread)
    games = [g for g in games if 'spread' in g and g['spread'] is not None]
    for game in games:
        game['margin'] = game['away_points'] - game['home_points']
    df = pd.DataFrame.from_records(games).dropna()
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