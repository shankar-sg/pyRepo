import numpy as np

def getMatrix(x,y,rate,direction):
     
    if direction == (1,0):
        nexts = [(x+rate,y),(x+rate,y-rate),(x+rate,y+rate)]
    elif direction == (-1,0):
        nexts = [(x-rate,y),(x-rate,y-rate),(x-rate,y+rate)]
    elif direction == (0,-1):
        nexts = [(x,y-rate),(x-rate,y-rate),(x+rate,y-rate)]
    elif direction == (0,1):
        nexts = [(x,y+rate),(x-rate,y+rate),(x+rate,y+rate)]
    elif direction == (1,1):
        nexts = [(x+rate,y+rate),(x+rate,y),(x+rate,y-rate),(x,y+rate),(x-rate,y+rate)]
    elif direction == (-1,-1):
        nexts = [(x-rate,y-rate),(x-rate,y),(x-rate,y+rate),(x,y-rate),(x+rate,y-rate)]
    elif direction == (-1,1):
        nexts = [(x-rate,y+rate),(x-rate,y),(x-rate,y-rate),(x,y+rate),(x+rate,y+rate)]
    elif direction == (1,-1):
        nexts = [(x+rate,y+rate),(x+rate,y),(x+rate,y-rate),(x,y-rate),(x-rate,y-rate)]
    elif direction == (0,0):
        nexts = [(x+rate,y+rate),(x+rate,y),(x+rate,y-rate),(x-rate,y+rate),(x-rate,y),(x-rate,y-rate),(x,y+rate),(x,y-rate)]
    
    returning = []
    for point in nexts:
        delx = point[0] - x
        dely = point[1] - y
        dirn = (int(delx/rate), int(dely/rate))
        returning.append((point[0], point[1], dirn))
        
    return tuple(returning)


def getLinearRegression(X,y): 
    direction, lastDirection = (0,0),(0,0) 
    slopeTemp, interceptTemp = 0,0 
    learn = 0.001
    minLoss = 99999999999999
    slope, intercept = 1,1 

    for _ in range(10000):
        deltaPoints = getMatrix(slope,intercept,learn,direction)     
        noNewMin = True
        for point in deltaPoints:
            xp,yp,dirn  = point
            loss = getLoss(xp,yp,X,y)
            #
            # Check for new lows
            #
            if loss < minLoss:
                #
                # Found a loss lower than current minimum ... 
                # We now need to move there next and look around
                #
                minLoss = loss
                slopeTemp = xp
                interceptTemp = yp
                lastDirection = dirn
                noNewMin = False           
        #
        # No new minimum has been found... This could be a local minimum
        #
        if (noNewMin):         
            break
        #
        # Move the reference point to the new minimum
        #
        slope = slopeTemp
        intercept = interceptTemp
        direction = lastDirection

    return slope,intercept


def getLoss(m,c,X,y):
    currLoss = (m*X + c - y ) ** 2    
    return sum(currLoss)

