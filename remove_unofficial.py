import os
from os import listdir
from os.path import isfile, join
import pandas as pd

inputDataPath = 'D:/Programs/python/ir_ratings_tf/data/'

# removes all rookie of unofficial races

fileNames = [f for f in listdir(inputDataPath) if isfile(join(inputDataPath, f))]

i = 0
for f in fileNames:
    session = pd.read_csv(inputDataPath + f, header=0, delimiter=';', encoding='latin1')
    ratings = session[['rating_old']].to_numpy()
    ratingChanges = session[['rating_delta']].to_numpy()

    if ratingChanges[0] == 0 or -1 in ratings:
        os.remove(inputDataPath + f)
        print('removed ' + f)

    i += 1