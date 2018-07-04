import pandas as pd
# FIFA TEAM STATS FOR WC 2018 upto R16
df = pd.read_csv('fifa_team_stats.csv')
# MATCHES RESULTS IN WC 2018 upto R16
matches = pd.read_csv('ALL_MATCHES.csv')

# Normailizing
matches['goals_scored_1']=matches['goals_scored_1']/matches['matches_played_1']

matches['goals_scored_2']=matches['goals_scored_2']/matches['matches_played_2']

matches['goals_against_2']=matches['goals_against_2']/matches['matches_played_2']

matches['attempts_on_target_2']=matches['attempts_on_target_2']/matches['matches_played_2']

matches['shots_attempted_2']=matches['shots_attempted_2']/matches['matches_played_2']

matches['attempts_off_target_2']=matches['attempts_off_target_2']/matches['matches_played_2']

matches['shots_blocked_2']=matches['shots_blocked_2']/matches['matches_played_2']

matches['goals_against_1']=matches['goals_against_1']/matches['matches_played_1']

matches['attempts_on_target_1']=matches['attempts_on_target_1']/matches['matches_played_1']

matches['shots_attempted_1']=matches['shots_attempted_1']/matches['matches_played_1']

matches['attempts_off_target_1']=matches['attempts_off_target_1']/matches['matches_played_1']

matches['shots_blocked_1']=matches['shots_blocked_1']/matches['matches_played_1']

matches['diff_goals_scored']=matches['goals_scored_1']-matches['goals_scored_2']

matches['diff_goals_against']=matches['goals_against_1']-matches['goals_against_2']

matches['diff_attempts_on_target']=matches['attempts_on_target_1']-matches['attempts_on_target_2']

matches['diff_shots_attempted']=matches['shots_attempted_1']-matches['shots_attempted_2']

matches['diff_attempts_off_target']=matches['attempts_off_target_1']-matches['attempts_off_target_2']

matches['diff_shots_blocked']=matches['shots_blocked_1']-matches['shots_blocked_2']

columns_of_interest = ['diff_goals_scored','diff_goals_against','diff_attempts_on_target','diff_shots_attempted','diff_attempts_off_target','diff_shots_blocked']    

train_X = matches[columns_of_interest]

train_Y = matches['results']

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

train_X[train_X.isnull()]
train_X=train_X.dropna()

rf.fit(train_X,train_Y)

def winning_probability (team_1,team_2):
    test_matches = pd.DataFrame()
    
    for i in range(len(team_1)):
        first = pd.DataFrame(df[df['Team']==team_1[i]])
        second = pd.DataFrame(df[df['Team']==team_2[i]])
        first = first.reset_index(drop=True)
        second = second.reset_index(drop=True)
        res = pd.DataFrame()

        for col in df.columns:
            res[col+"_1"]=pd.DataFrame(first[col])
            res[col+"_2"]=pd.DataFrame(second[col])
        for col in df.columns:
            res[col+"_2"]=pd.DataFrame(second[col])
            res[col+"_1"]=pd.DataFrame(first[col])
			
        test_matches = test_matches.append(res)
        
        test_matches['diff_goals_scored']=test_matches['goals_scored_1']-test_matches['goals_scored_2']
    
        test_matches['diff_goals_against']=test_matches['goals_against_1']-test_matches['goals_against_2']
    
        test_matches['diff_attempts_on_target']=test_matches['attempts_on_target_1']-test_matches['attempts_on_target_2']
    
        test_matches['diff_shots_attempted']=test_matches['shots_attempted_1']-test_matches['shots_attempted_2']
       
        test_matches['diff_attempts_off_target']=test_matches['attempts_off_target_1']-test_matches['attempts_off_target_2']
    
        test_matches['diff_shots_blocked']=test_matches['shots_blocked_1']-test_matches['shots_blocked_2']
    
        test_X = test_matches[columns_of_interest]
    
        test_X[test_X.isnull()]
    
        test_X=test_X.dropna()
        pred_Y = rf.predict(test_X)
        return pred_Y[0]

# Quarter Finals Winning Probabilites: #
print ('France vs Uruguay -> %f' %(winning_probability(['France'],['Uruguay'])))
print ('Belgium vs Brazil -> %f' %(winning_probability(['Belgium'],['Brazil'])))
print ('Sweden vs England -> %f' %(winning_probability(['Sweden'],['England'])))
print ('Russia vs Croatia -> %f' %(winning_probability(['Russia'],['Croatia'])))


