import pandas as data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
#from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Path of the file to read
realEstate = pd.read_csv("realEstate.csv")
realEstate.shape  #414, 8