import numpy as np
from random import randint
import matplotlib.pyplot as plt

def shuffle(inputarray):
    '''Takes an array and shuffles the order '''
    array=inputarray.copy()
    outputarr = []
    for i in range(len(array)):
        x = randint(0,len(array)-1)
        outputarr.append(array[x])
        array.pop(x)
    return(outputarr)

def initgame(Chars,Cards):
    '''Initializes the game by randomising the order of players and cards '''
    players = shuffle(Chars)
    #players
    cards = shuffle(Cards)
    return(players,cards)

def dealcard(deck):
    return(deck.pop(0))

def deal2card(deck):
    '''Takes a deck of cards and returns the first two elements, deletes them from the list '''
    return([dealcard(deck),dealcard(deck)])

def dealround(deck,playerlist):
    '''Returns two cards for each player '''
    hands = []
    for i in range(len(playerlist)):
        hands.append([playerlist[i],deal2card(deck)])
    return(hands)

## Number of people with correct resource:
def Nrightresource(Xhands,resource):
    '''returns the number of players who have right resource
    and the number of marines who have the right resource
    Any pair (except SLIME) is a correct resource
    '''
    count = 0
    goodcount = 0
    for hand in Xhands:
        for i in range(2):
            if hand[1][0]==hand[1][1] and hand[1][0]!=resource and hand[1][0]!='SLIME':
                #print(hand[1][0],hand[1][1])
                if hand[0] == 'Marine':
                    goodcount+=1
                count+=1
                break
            if hand[1][i] == resource or hand[1][i] == 'Printable':
                if hand[0] == 'Marine':
                    goodcount+=1
                count+=1
                break
    return(count,goodcount)

def round(missions):
    '''This function creates a round by randomly selecting missions'''
    phase = []
    for i in range(2):
        randnum=randint(0,len(missions)-1)
        phase.append(missions[randnum])
        missions.pop(randnum)
    return(phase)

## Missions are tuples that are structured as
## Name , cost , Primary resource , Secondary resource , or/and , damage , movement
def makemissbank():
    mission1 = ('Hull Repair',3,'Tool Kit',np.NaN,-1,1)
    mission2 = ('Life Support',3,'Electronics',np.NaN,-1,1)
    mission3 = ('Refuel',3,'Fuel Cell',np.NaN,-1,1)
    mission4 = ('Lighten the Load',2,'Tool Kit',np.NaN,-2,1)
    mission5 = ('Upgrade Comms',2,'Electronics',np.NaN,-2,1)
    phase1missionbank=[mission1,mission2,mission3,mission4,mission5]
    return(phase1missionbank)

def MissionPos(Xhands,Mission,printer=False):
    '''return if the mission is possible'''
    if Nrightresource(Xhands,Mission[2])[1]>=Mission[1]:
        if printer == True:
            print('Aliens ', Nrightresource(Xhands,Mission[2])[0]-Nrightresource(Xhands,Mission[2])[1],' Marines resources :', Nrightresource(Xhands,Mission[2])[1],' Required :', Mission[1])
        return(True)
    else:
        if printer == True:
            print('Aliens ', Nrightresource(Xhands,Mission[2])[0]-Nrightresource(Xhands,Mission[2])[1],' Marines resources :', Nrightresource(Xhands,Mission[2])[1],' Required :', Mission[1])
        return(False)

## probability of success
def probofsuccess(XHands,phayse1,phayse2):
    '''returns the naive probability of success'''
    if MissionPos(XHands,phayse1):
        return(Nrightresource(XHands,phayse1[2])[1]/(Nrightresource(XHands,phayse1[2])[0]))
    else:
        #print('mission impossible')
        return(0)

# Parameters for the game

playernum = 6
aliens = int(np.floor(6/3))
characters = (playernum-aliens)*['Marine']+aliens*['Alien']
characters
cardlist = (playernum)*(['Tool Kit'] + ['Fuel Cell'] + ['Electronics']) +2*['Printable']+2*['SLIME']
players,cards = initgame(characters,cardlist)
hands = dealround(cards,players)
phase1missionbank=makemissbank()
Phase1 = round(phase1missionbank)

results=[]
for i in range(100):
    players,cards = initgame(characters,cardlist)
    hands = dealround(cards,players)
    phase1missionbank=makemissbank()
    Phase1 = round(phase1missionbank)
    prob1 = probofsuccess(hands,Phase1[0],0)
    prob2 = probofsuccess(hands,Phase1[1],0)
    results.append(prob1+(1-prob1)*prob2)
plt.hist(results)
np.mean(results)

#PCompleteMission =
## probability of failure
