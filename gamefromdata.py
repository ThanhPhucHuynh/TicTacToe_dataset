import numpy as np
import pandas as pd

from sklearn.tree import plot_tree

#read data from file csv
df = pd.read_csv('tic-tac-toe.data',
        names = ["V1", "V2", "V3", "V4", "V5", "V6", 
                                    "V7", "V8", "V9", "V10"],
                                    sep=",")

#print(df): raw data
#     V1 V2 V3 V4 V5 V6 V7 V8 V9       V10
# 0    x  x  x  x  o  o  x  o  o  positive
# 1    x  x  x  x  o  o  o  x  o  positive
# 2    x  x  x  x  o  o  o  o  x  positive
# 3    x  x  x  x  o  o  o  b  b  positive
# 4    x  x  x  x  o  o  b  o  b  positive
# ..  .. .. .. .. .. .. .. .. ..       ...
# 953  o  x  x  x  o  o  o  x  x  negative
# 954  o  x  o  x  x  o  x  o  x  negative
# 955  o  x  o  x  o  x  x  o  x  negative
# 956  o  x  o  o  x  x  x  o  x  negative
# 957  o  o  x  x  x  o  o  x  x  negative


df['V1'],v1 = pd.factorize(df['V1'], sort=True)
df['V2'],v2 = pd.factorize(df['V2'], sort=True)
df['V3'],v3 = pd.factorize(df['V3'], sort=True)
df['V4'],v4 = pd.factorize(df['V4'], sort=True)
df['V5'],v5 = pd.factorize(df['V5'], sort=True)
df['V6'],v6 = pd.factorize(df['V6'], sort=True)
df['V7'],v7 = pd.factorize(df['V7'], sort=True)
df['V8'],v8 = pd.factorize(df['V8'], sort=True)
df['V9'],v9 = pd.factorize(df['V9'], sort=True)
df['V10'],v10 = pd.factorize(df['V10'], sort=True)

# Exchange data with {"x":2,"y":1,"b":0 }
#with class lables {"positive":1,"nagative":2 }
#data after exchange
#       V1  V2  V3  V4  V5  V6  V7  V8  V9  V10
# 0     2   2   2   2   1   1   2   1   1    1
# 1     2   2   2   2   1   1   1   2   1    1
# 2     2   2   2   2   1   1   1   1   2    1
# 3     2   2   2   2   1   1   1   0   0    1
# 4     2   2   2   2   1   1   0   1   0    1
# ..   ..  ..  ..  ..  ..  ..  ..  ..  ..  ...
# 953   1   2   2   2   1   1   1   2   2    0
# 954   1   2   1   2   2   1   2   1   2    0
# 955   1   2   1   2   1   2   2   1   2    0
# 956   1   2   1   1   2   2   2   1   2    0
# 957   1   1   2   2   2   1   1   2   2    0


#print("data sao : \n",df)

className = [v10[0],v10[1]]
# print( className)
# ['negative', 'positive']

# print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 958 entries, 0 to 957
# Data columns (total 10 columns):
# V1     958 non-null int64
# V2     958 non-null int64
# V3     958 non-null int64
# V4     958 non-null int64
# V5     958 non-null int64
# V6     958 non-null int64
# V7     958 non-null int64
# V8     958 non-null int64
# V9     958 non-null int64
# V10    958 non-null int64
# dtypes: int64(10)
# memory usage: 75.0 KB

# print(df.describe())
#                V1          V2          V3          V4  ...          V7          V8          V9         V10
# count  958.000000  958.000000  958.000000  958.000000  ...  958.000000  958.000000  958.000000  958.000000
# mean     1.222338    1.133612    1.222338    1.133612  ...    1.222338    1.133612    1.222338    0.653445
# std      0.775569    0.798966    0.775569    0.798966  ...    0.775569    0.798966    0.775569    0.476121
# min      0.000000    0.000000    0.000000    0.000000  ...    0.000000    0.000000    0.000000    0.000000
# 25%      1.000000    0.000000    1.000000    0.000000  ...    1.000000    0.000000    1.000000    0.000000
# 50%      1.000000    1.000000    1.000000    1.000000  ...    1.000000    1.000000    1.000000    1.000000
# 75%      2.000000    2.000000    2.000000    2.000000  ...    2.000000    2.000000    2.000000    1.000000
# max      2.000000    2.000000    2.000000    2.000000  ...    2.000000    2.000000    2.000000    1.000000

# [8 rows x 10 columns]

#Get attributes and labels
feature_names = ['V1','V2','V3','V4', 'V5', 'V6', 'V7', 'V8', 'V9']
x = df[feature_names] # Features
y = df['V10'] # lables
# print(x,y)


 
from IPython.display import Image
from io import StringIO
from sklearn import tree
import pydotplus
#plot the decision tree and draw it o save it to .png
#use can use to function pot tree for sklen
def plot_decision_tree(clf, features, classes):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("iris.png")#thii
    return Image(graph.create_png())






print("\nDecisionTreeClassifier")

from sklearn.tree import DecisionTreeClassifier 
# clf = DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(x,y)

# print(clf)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5,shuffle=True)

# print(x_train,y_train,x_test,y_test)

# clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=80) # change this classifier and check the impact
clf = DecisionTreeClassifier(criterion="entropy")
clf = clf.fit(x_train,y_train)
plot_decision_tree(clf, feature_names, className)

from sklearn import metrics
from sklearn.metrics import classification_report


# use the model to make predictions with the test data
y_pred = clf.predict(x_test)
# how did our model perform?
count_misclassified = (y_test != y_pred).sum()
# print('Misclassified samples (Mau phan loai sai): {}'.format(count_misclassified))
accuracy = metrics.accuracy_score(y_test, y_pred)
# print('Accuracy: {:.2f}'.format(accuracy))
# print("DTC report: \n",classification_report(y_test,y_pred))


print("Anticipate: [2, 2, 1, 2, 1, 0, 1, 0, 0] \nAnd         [2, 2, 2, 1, 1, 0, 1, 0, 0]")

negative_test = np.array ([2, 2, 1, 2, 1, 0, 1, 0, 0])
positive_test = np.array ([2, 2, 2, 1, 1, 0, 1, 0, 0])
test_group = [negative_test, positive_test]
y_pred = clf.predict(test_group)

def countDuDoan(a):
    y_pred = clf.predict(a)
    print(y_pred)
    return y_pred

from math import inf as infinity

print(infinity)
game_state = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
players = ['X','O']

def play_move(state, player, block_num):
    if state[int((block_num-1)/3)][(block_num-1)%3] is ' ':
        state[int((block_num-1)/3)][(block_num-1)%3] = player
    else:
        block_num = int(input("Block is not empty, ya blockhead! Choose again: "))
        play_move(state, player, block_num)
    
def copy_game_state(state):
    new_state = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]
    for i in range(3):
        for j in range(3):
            new_state[i][j] = state[i][j]
    return new_state
    
def check_current_state(game_state):
    # Check if draw
    draw_flag = 0
    for i in range(3):
        for j in range(3):
            if game_state[i][j] is ' ':
                draw_flag = 1
    if draw_flag is 0:
        return None, "Draw"
    
    # Check horizontals
    if (game_state[0][0] == game_state[0][1] and game_state[0][1] == game_state[0][2] and game_state[0][0] is not ' '):
        return game_state[0][0], "Done"
    if (game_state[1][0] == game_state[1][1] and game_state[1][1] == game_state[1][2] and game_state[1][0] is not ' '):
        return game_state[1][0], "Done"
    if (game_state[2][0] == game_state[2][1] and game_state[2][1] == game_state[2][2] and game_state[2][0] is not ' '):
        return game_state[2][0], "Done"
    
    # Check verticals
    if (game_state[0][0] == game_state[1][0] and game_state[1][0] == game_state[2][0] and game_state[0][0] is not ' '):
        return game_state[0][0], "Done"
    if (game_state[0][1] == game_state[1][1] and game_state[1][1] == game_state[2][1] and game_state[0][1] is not ' '):
        return game_state[0][1], "Done"
    if (game_state[0][2] == game_state[1][2] and game_state[1][2] == game_state[2][2] and game_state[0][2] is not ' '):
        return game_state[0][2], "Done"
    
    # Check diagonals
    if (game_state[0][0] == game_state[1][1] and game_state[1][1] == game_state[2][2] and game_state[0][0] is not ' '):
        return game_state[1][1], "Done"
    if (game_state[2][0] == game_state[1][1] and game_state[1][1] == game_state[0][2] and game_state[2][0] is not ' '):
        return game_state[1][1], "Done"
    
    return None, "Not Done"

def print_board(game_state):
    print('----------------')
    print('| ' + str(game_state[0][0]) + ' || ' + str(game_state[0][1]) + ' || ' + str(game_state[0][2]) + ' |')
    print('----------------')
    print('| ' + str(game_state[1][0]) + ' || ' + str(game_state[1][1]) + ' || ' + str(game_state[1][2]) + ' |')
    print('----------------')
    print('| ' + str(game_state[2][0]) + ' || ' + str(game_state[2][1]) + ' || ' + str(game_state[2][2]) + ' |')
    print('----------------')
    
    
def getBestMove(state, player):
    '''
    Minimax Algorithm
    '''
    winner_loser , done = check_current_state(state)
    if done == "Done" and winner_loser == 'O': # If AI won
        return 1
    elif done == "Done" and winner_loser == 'X': # If Human won
        return -1
    elif done == "Draw":    # Draw condition
        return 0
        
    moves = []
    empty_cells = []
    a = []
    best_move = None
    for i in range(3):
        for j in range(3):
            if state[i][j] is ' ':
                empty_cells.append(i*3 + (j+1))
    
    for i in range(3):
        for j in range(3):
            if state[i][j] is ' ':
                a.append(2)
            elif state[i][j] is 'X':
                a.append(1)
            else:
                a.append(0)
    (clf.predict([np.array(a)])[0])

            
    return best_move

# PLaying
play_again = 'Y'
while play_again == 'Y' or play_again == 'y':
    game_state = [[' ',' ',' '],
              [' ',' ',' '],
              [' ',' ',' ']]
    current_state = "Not Done"
    print("\nNew Game!")
    print_board(game_state)
    player_choice = input("Choose which player goes first - X (You - the petty human) or O(The mighty AI): ")
    winner = None
    
    if player_choice == 'X' or player_choice == 'x':
        current_player_idx = 0
    else:
        current_player_idx = 1
        
    while current_state == "Not Done":
        if current_player_idx == 0: # Human's turn
            block_choice = int(input("Oye Human, your turn! Choose where to place (1 to 9): "))
            play_move(game_state ,players[current_player_idx], block_choice)
        else:   # AI's turn
            block_choice = getBestMove(game_state, players[current_player_idx])
            play_move(game_state ,players[current_player_idx], block_choice)
            print("AI plays move: " + str(block_choice))
        print_board(game_state)
        winner, current_state = check_current_state(game_state)
        if winner is not None:
            print(str(winner) + " won!")
        else:
            current_player_idx = (current_player_idx + 1)%2
        
        if current_state is "Draw":
            print("Draw!")
            
    play_again = input('Wanna try again BIYTACH?(Y/N) : ')
    if play_again == 'N':
        print('Suit yourself bitch!')

