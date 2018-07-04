
# coding: utf-8

# In[14]:


import pandas as pd
# FIFA TEAM STATS FOR WC 2018 upto R16
df = pd.read_csv('fifa_team_stats.csv')
# MATCHES RESULTS IN WC 2018 upto R16
matches = pd.read_csv('ALL_MATCHES.csv')
matches.head()


# In[15]:



# Normailizing
matches['goals_scored_1']=matches['goals_scored_1']/matches['matches_played_1']

matches['goals_scored_2']=matches['goals_scored_2']/matches['matches_played_2']

matches['goals_against_2']=matches['goals_against_2']/matches['matches_played_2']

matches['attempts_on_target_2']=matches['attempts_on_target_2']/matches['matches_played_2']

matches['shots_attempted_2']=matches['shots_attempted_2']/matches['matches_played_2']

matches['attempts_off_target_2']=matches['attempts_off_target_2']/matches['matches_played_2']

matches['shots_blocked_2']=matches['shots_blocked_2']/matches['matches_played_2']

matches['clearances_2']=matches['clearances_2']/matches['matches_played_2']

matches['goals_against_1']=matches['goals_against_1']/matches['matches_played_1']

matches['attempts_on_target_1']=matches['attempts_on_target_1']/matches['matches_played_1']

matches['shots_attempted_1']=matches['shots_attempted_1']/matches['matches_played_1']

matches['attempts_off_target_1']=matches['attempts_off_target_1']/matches['matches_played_1']

matches['shots_blocked_1']=matches['shots_blocked_1']/matches['matches_played_1']

matches['clearances_1']=matches['clearances_1']/matches['matches_played_1']

matches.head()
matches_2 = pd.DataFrame()
visited=[]
for col in matches.columns:
    c_name = str(col)
    if ('result' not in c_name):
        col_names = c_name.split('_')
        y = len(col_names)
        c_name_2 = ""
        for j in range(y-1):
            if (c_name_2):
                c_name_2 = c_name_2 + "_"
            c_name_2=c_name_2 + col_names[j]
        if (c_name_2 not in visited):
            matches_2[c_name_2+"_1"]=matches[col]
            visited.append(c_name_2)
        else:
            matches_2[c_name_2+"_2"]=matches[col]
    else:
        matches_2['results']=1-matches['results']
matches_2.head()
matches=matches.append(matches_2)
matches = matches.reset_index(drop=True)
matches[matches['Team_1']=='Russia']


# In[16]:


#Difference in Opposing Teams
matches['diff_goals_scored']=matches['goals_scored_1']-matches['goals_scored_2']

matches['diff_goals_against']=matches['goals_against_1']-matches['goals_against_2']

matches['diff_attempts_on_target']=matches['attempts_on_target_1']-matches['attempts_on_target_2']

matches['diff_shots_attempted']=matches['shots_attempted_1']-matches['shots_attempted_2']

matches['diff_attempts_off_target']=matches['attempts_off_target_1']-matches['attempts_off_target_2']

matches['diff_shots_blocked']=matches['shots_blocked_1']-matches['shots_blocked_2']

matches['diff_clearances']=matches['clearances_1']-matches['clearances_2']

matches.head()


# In[17]:


#Training Data
columns_of_interest = ['diff_goals_scored','diff_goals_against','diff_attempts_on_target','diff_shots_attempted','diff_attempts_off_target','diff_shots_blocked','diff_clearances']    

train_X = matches[columns_of_interest]

train_Y = matches['results']

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

train_X[train_X.isnull()]
train_X=train_X.dropna()

rf.fit(train_X,train_Y)


# In[18]:


#Function to return weight of wining of team_1 against team_2

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
			
        test_matches = test_matches.append(res)
        
        test_matches['goals_scored_1']=test_matches['goals_scored_1']/test_matches['matches_played_1']

        test_matches['goals_scored_2']=test_matches['goals_scored_2']/test_matches['matches_played_2']

        test_matches['goals_against_2']=test_matches['goals_against_2']/test_matches['matches_played_2']

        test_matches['attempts_on_target_2']=test_matches['attempts_on_target_2']/test_matches['matches_played_2']

        test_matches['shots_attempted_2']=test_matches['shots_attempted_2']/test_matches['matches_played_2']

        test_matches['attempts_off_target_2']=test_matches['attempts_off_target_2']/test_matches['matches_played_2']

        test_matches['shots_blocked_2']=test_matches['shots_blocked_2']/test_matches['matches_played_2']

        test_matches['clearances_2']=test_matches['clearances_2']/test_matches['matches_played_2']

        test_matches['goals_against_1']=test_matches['goals_against_1']/test_matches['matches_played_1']

        test_matches['attempts_on_target_1']=test_matches['attempts_on_target_1']/test_matches['matches_played_1']

        test_matches['shots_attempted_1']=test_matches['shots_attempted_1']/test_matches['matches_played_1']

        test_matches['attempts_off_target_1']=test_matches['attempts_off_target_1']/test_matches['matches_played_1']

        test_matches['shots_blocked_1']=test_matches['shots_blocked_1']/test_matches['matches_played_1']

        test_matches['clearances_1']=test_matches['clearances_1']/test_matches['matches_played_1']

        test_matches['diff_goals_scored']=test_matches['goals_scored_1']-test_matches['goals_scored_2']
    
        test_matches['diff_goals_against']=test_matches['goals_against_1']-test_matches['goals_against_2']
    
        test_matches['diff_attempts_on_target']=test_matches['attempts_on_target_1']-test_matches['attempts_on_target_2']
    
        test_matches['diff_shots_attempted']=test_matches['shots_attempted_1']-test_matches['shots_attempted_2']
       
        test_matches['diff_attempts_off_target']=test_matches['attempts_off_target_1']-test_matches['attempts_off_target_2']
    
        test_matches['diff_shots_blocked']=test_matches['shots_blocked_1']-test_matches['shots_blocked_2']
        
        test_matches['diff_clearances']=matches['clearances_1']-matches['clearances_2']
    
        test_X = test_matches[columns_of_interest]
    
        test_X[test_X.isnull()]
    
        test_X=test_X.dropna()
        pred_Y = rf.predict(test_X)
        return pred_Y[0]

#Function to return probability of winning of Team 1 against Team 2

def get_probability (team_1,team_2):
    X1 = winning_probability(team_1,team_2)
    X2 = winning_probability(team_2,team_1)
    X = X1/(X1+X2)
    return X


# In[21]:


# Quarter Finals Winning Probabilites: #
print ('Uruguay vs France -> %f' %(get_probability(['Uruguay'],['France'])))
print ('Belgium vs Brazil -> %f' %(get_probability(['Belgium'],['Brazil'])))
print ('England vs Sweden -> %f' %(get_probability(['England'],['Sweden'])))
print ('Croatia vs Russia -> %f' %(get_probability(['Croatia'],['Russia'])))

