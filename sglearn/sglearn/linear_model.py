#
# Linear Regression model implemented in sglearn package
#
class LinearRegression:
    def __init__(self):
        self.coef_ = 0
        self.intercept_ = 0

    def predict(self, X):
        return X*self.coef_ + self.intercept_

    #
    # Internal method to tell what are the next Points to search for the minimum Loss
    # Ideally, 9 points with the current point as centre must be searched for the loss
    # Since the previous check removed a few of the 9 points, this is an optimization
    # to give only new Points where loss is not yet calculated
    #

    def getMatrix(self, x, y, rate, direction):
        if direction == (1, 0):
            nexts = [(x + rate, y), (x + rate, y - rate), (x + rate, y + rate)]
        elif direction == (-1, 0):
            nexts = [(x - rate, y), (x - rate, y - rate), (x - rate, y + rate)]
        elif direction == (0, -1):
            nexts = [(x, y - rate), (x - rate, y - rate), (x + rate, y - rate)]
        elif direction == (0, 1):
            nexts = [(x, y + rate), (x - rate, y + rate), (x + rate, y + rate)]
        elif direction == (1, 1):
            nexts = [(x + rate, y + rate), (x + rate, y), (x + rate, y - rate), (x, y + rate), (x - rate, y + rate)]
        elif direction == (-1, -1):
            nexts = [(x - rate, y - rate), (x - rate, y), (x - rate, y + rate), (x, y - rate), (x + rate, y - rate)]
        elif direction == (-1, 1):
            nexts = [(x - rate, y + rate), (x - rate, y), (x - rate, y - rate), (x, y + rate), (x + rate, y + rate)]
        elif direction == (1, -1):
            nexts = [(x + rate, y + rate), (x + rate, y), (x + rate, y - rate), (x, y - rate), (x - rate, y - rate)]
        elif direction == (0, 0):
            nexts = [(x + rate, y + rate), (x + rate, y), (x + rate, y - rate), (x - rate, y + rate), (x - rate, y),
                     (x - rate, y - rate), (x, y + rate), (x, y - rate)]

        returning = []
        for point in nexts:
            delx = point[0] - x
            dely = point[1] - y
            dirn = (int(delx / rate), int(dely / rate))
            returning.append((point[0], point[1], dirn))

        return tuple(returning)

    #
    # Takes in a 1D array and returns a 1D array
    # Standard models need X to be reshaped as a 2D array
    # Fits and creates the linear model coefficient and intercept
    #
    def fit(self, X, y):
        direction, lastDirection = (0, 0), (0, 0)
        slopeTemp, interceptTemp = 0, 0
        learn = 0.001
        minLoss = 99999999999999
        slope, intercept = 1, 1

        for _ in range(10000):
            deltaPoints = self.getMatrix(slope, intercept, learn, direction)
            noNewMin = True
            for point in deltaPoints:
                xp, yp, dirn = point
                loss = self.getLoss(xp, yp, X, y)
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
            if noNewMin:
                break
            #
            # Move the reference point to the new minimum
            #
            slope = slopeTemp
            intercept = interceptTemp
            direction = lastDirection

        self.coef_ ,self.intercept_ = slope, intercept

    #
    # Calculates loss of the data for a given pair of intercept & coefficient.
    # This function is called repeatedly till the local minima/ maximum loss is found
    #
    def getLoss(self, m, c, X, y):
        currLoss = (m * X + c - y) ** 2
        return sum(currLoss)


