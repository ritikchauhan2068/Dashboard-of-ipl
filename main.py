import streamlit as st
import pandas as pd
import numpy as np
import json
# Add custom styles to your Streamlit app


# Load IPL match data and ball-by-ball data
matches = pd.read_csv('IPL_Matches_2008_2022 - IPL_Matches_2008_2022 (2).csv')
balls = pd.read_csv('IPL_Ball_by_Ball_2008_2022 - IPL_Ball_by_Ball_2008_2022.csv')

# Merge match and ball data
df = balls.merge(matches, on='ID', how='inner')

# Create a column for BowlingTeam
df['BowlingTeam'] = df.Team1 + df.Team2
df['BowlingTeam'] = df[['BowlingTeam', 'BattingTeam']].apply(lambda x: x.values[0].replace(x.values[1], ''), axis=1)

# Create batter_data with required columns
batter_data = df[np.append(balls.columns.values, ['BowlingTeam', 'Player_of_Match'])]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def season():
    seasons=list(matches['Season'].unique())
    seasonp={
        'seasons' : seasons
    }
    return seasonp

def teamsApi():
    teams = list(set(list(matches['Team1']) + list(matches['Team2'])))
    team_dict = {
        'teams': teams
    }
    return team_dict

def team_vs_team(team1, team2):
    temp_df = matches[(matches['Team1'] == team1) & (matches['Team2'] == team2) | (matches['Team1'] == team2) & (
                matches['Team2'] == team1)]
    total_matches = temp_df.shape[0]
    winning_team1 = temp_df['WinningTeam'].value_counts()[team1]
    winning_team2 = temp_df['WinningTeam'].value_counts()[team2]
    total_winning_matches = winning_team1 + winning_team2
    draw = total_matches - total_winning_matches
    response = {
        'total_matches': str(total_matches),
        team1: str(winning_team1),
        team2: str(winning_team2),
        'draw': str(draw)
    }
    return response

def allRecord(team):
    df = matches[(matches['Team1'] == team) | (matches['Team2'] == team)].copy()
    matches_played = df.shape[0]
    won = df[df.WinningTeam == team].shape[0]
    nr = df[df.WinningTeam.isnull()].shape[0]
    loss = matches_played - won - nr
    nt = df[(df.MatchNumber == 'Final') & (df.WinningTeam == team)].shape[0]
    return {
        'matchesplayed': matches_played,
        'won': won,
        'loss': loss,
        'noResult': nr,
        'title': nt
    }

def batsman_record(batsman, df):
    if df.empty:
        return np.nan
    out = df[df['player_out'] == batsman].shape[0]
    df = df[df['batter'] == batsman]
    fours = df[df['batsman_run'] == 4].shape[0]
    sixes = df[df['batsman_run'] == 6].shape[0]
    innings = df['ID'].unique().shape[0]
    runs = df['batsman_run'].sum()
    if out:
        avg = runs / out
    else:
        avg = np.inf
    no_balls = df[~(df['extra_type'] == 'wides')].shape[0]
    if no_balls:
        strike_rate = runs / no_balls * 100
    else:
        strike_rate = 0
    gb = df.groupby('ID').sum()
    fifties = gb[(gb['batsman_run'] >= 50) & (gb['batsman_run'] < 100)].shape[0]
    hundreds = gb[gb['batsman_run'] >= 100].shape[0]
    mom = df[df['Player_of_Match'] == batsman].drop_duplicates('ID', keep='first').shape[0]
    not_out = innings - out
    return {
        'innings': innings,
        'runs': runs,
        'fours': fours,
        'sixes': sixes,
        'avg': float(avg),
        'strikeRate': float(strike_rate),
        'fifties': fifties,
        'hundreds': hundreds,
        'notOut': not_out,
        'mom': mom,
        'number of balls': no_balls
    }

def batsmanVsTeam(batsman, team, df):
    df = df[df.BowlingTeam == team].copy()
    return batsman_record(batsman, df)

def batsmanAPI(batsman, balls=batter_data):
    df = balls[balls.innings.isin([1, 2])]
    self_record = batsman_record(batsman, df=df)
    TEAMS = matches.Team1.unique()
    against = {team: batsmanVsTeam(batsman, team, df) for team in TEAMS}
    data = {
        batsman: {'all': self_record, 'against': against}
    }
    return json.dumps(data, cls=NpEncoder)

def bowlerwicket(a):
    if a.iloc[0] in ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket']:
        return a.iloc[1]
    else:
        return 0

batter_data['isBowlerWicket'] = batter_data[['kind', 'isWicketDelivery']].apply(bowlerwicket, axis=1)

def bowlerrun(b):
    if b.iloc[0] in ['penalty', 'legbyes', 'byes']:
        return 0
    else:
        return b.iloc[1]

batter_data['bowler_run'] = batter_data[['extra_type', 'total_run']].apply(bowlerrun, axis=1)
batter_data['batsman_run'] = batter_data['total_run'] - batter_data['bowler_run']
batter_data['isBatsmanWicket'] = batter_data[['kind', 'isWicketDelivery']].apply(lambda x: x[0] if x[1] else '', axis=1)

# Set the background image
background_image = 'background_image.jpg'
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url({background_image});
        background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app starts here
st.title("IPL Data Analysis")
st.sidebar.title("Navigation")

# Sidebar for selecting options
selected_option = st.sidebar.selectbox("Select an option", ["Home", "Teams", "Team vs. Team", "Team Record", "Batting Record", "Bowling Record"])




if selected_option == "Home":
    st.header("Welcome to IPL Data Analysis")
    seaso= season()
    for i in seaso['seasons']:
        st.write(i)
    st.write("Select an option from the sidebar to explore IPL data.")

elif selected_option == "Teams":
    st.header("IPL Teams")
    teams = teamsApi()
    st.write("List of IPL teams:")
    for team in teams['teams']:
        st.write(team)

elif selected_option == "Team vs. Team":
    st.header("Team vs. Team Analysis")
    st.write("Compare two IPL teams:")
    team1 = st.selectbox("Select Team 1", teamsApi()['teams'])
    team2 = st.selectbox("Select Team 2", teamsApi()['teams'])

    if st.button("Compare"):
        response = team_vs_team(team1, team2)
        st.write(f"Total matches: {response['total_matches']}")
        st.write(f"{team1} wins: {response[team1]}")
        st.write(f"{team2} wins: {response[team2]}")
        st.write(f"Draw: {response['draw']}")

elif selected_option == "Team Record":
    st.header("Team Record Analysis")
    team_name = st.selectbox("Select a team", teamsApi()['teams'])

    if st.button("Get Team Record"):
        response = allRecord(team_name)
        st.write(f"Matches played: {response['matchesplayed']}")
        st.write(f"Won: {response['won']}")
        st.write(f"Loss: {response['loss']}")
        st.write(f"No Result: {response['noResult']}")
        st.write(f"Titles: {response['title']}")

elif selected_option == "Batting Record":
    st.header("Batting Record Analysis")
    batsman_name = st.text_input("Enter a batsman's name")

    if st.button("Get Batting Record"):
        response = batsmanAPI(batsman_name)
        data = json.loads(response)

        if batsman_name in data:
            st.write(f"Batting Record for {batsman_name}:")
            st.write("Overall Stats:")
            st.write(f"Total Innings: {data[batsman_name]['all']['innings']}")
            st.write(f"Total Runs: {data[batsman_name]['all']['runs']}")
            st.write(f"Total Fours: {data[batsman_name]['all']['fours']}")
            st.write(f"Total Sixes: {data[batsman_name]['all']['sixes']}")
            st.write(f"Batting Average: {data[batsman_name]['all']['avg']:.2f}")
            st.write(f"Strike Rate: {data[batsman_name]['all']['strikeRate']:.2f}")
            st.write(f"Fifties: {data[batsman_name]['all']['fifties']}")
            st.write(f"Hundreds: {data[batsman_name]['all']['hundreds']}")
            st.write(f"Man of the Match Awards: {data[batsman_name]['all']['mom']}")

            st.header("Stats Against Each Team:")
            for team, stats in data[batsman_name]['against'].items():
                st.write(f"Against {team}:")
                st.write(f"Total Innings: {stats['innings']}")
                st.write(f"Total Runs: {stats['runs']}")
                st.write(f"Total Fours: {stats['fours']}")
                st.write(f"Total Sixes: {stats['sixes']}")
                st.write(f"Batting Average: {stats['avg']:.2f}")
                st.write(f"Strike Rate: {stats['strikeRate']:.2f}")
                st.write(f"Fifties: {stats['fifties']}")
                st.write(f"Hundreds: {stats['hundreds']}")
                st.write(f"Man of the Match Awards: {stats['mom']}")

        else:
            st.write(f"No data found for {batsman_name}")

elif selected_option == "Bowling Record":
    st.header("Bowling Record Analysis")
    bowler_name = st.text_input("Enter a bowler's name")

    if st.button("Get Bowling Record"):
        response = bowlerAPI(bowler_name)
        data = json.loads(response)
        if bowler_name in data:
            st.write(f"Bowling Record for {bowler_name}:")
            st.write("Overall Stats:")
            st.write(f"Total Innings: {data[bowler_name]['all']['innings']}")
            st.write(f"Total Wickets: {data[bowler_name]['all']['wicket']}")
            st.write(f"Economy Rate: {data[bowler_name]['all']['economy']:.2f}")
            st.write(f"Bowling Average: {data[bowler_name]['all']['avg']:.2f}")
            st.write(f"Strike Rate: {data[bowler_name]['all']['strikeRate']:.2f}")
            st.write(f"Fours Conceded: {data[bowler_name]['all']['fours']}")
            st.write(f"Sixes Conceded: {data[bowler_name]['all']['sixes']}")
            st.write(f"Best Bowling Figure: {data[bowler_name]['all']['best_figure']}")
            st.write(f"3+ Wickets in an Innings: {data[bowler_name]['all']['3+W']}")
            st.write(f"Man of the Match Awards: {data[bowler_name]['all']['mom']}")

            st.write("Stats Against Each Team:")
            for team, stats in data[bowler_name]['against'].items():
                st.write(f"Bowling Record Against {team}:")
                st.write(f"Total Innings: {stats['innings']}")
                st.write(f"Total Wickets: {stats['wicket']}")
                st.write(f"Economy Rate: {stats['economy']:.2f}")
                st.write(f"Bowling Average: {stats['avg']:.2f}")
                st.write(f"Strike Rate: {stats['strikeRate']:.2f}")
                st.write(f"Fours Conceded: {stats['fours']}")
                st.write(f"Sixes Conceded: {stats['sixes']}")
                st.write(f"Best Bowling Figure: {stats['best_figure']}")
                st.write(f"3+ Wickets in an Innings: {stats['3+W']}")
                st.write(f"Man of the Match Awards: {stats['mom']}")

        else:
            st.write(f"No data found for {bowler_name}")
