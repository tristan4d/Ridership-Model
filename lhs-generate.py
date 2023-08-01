import pandas as pd
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import *

# Read in data and perform some clean up

# df = pd.read_csv('../../../Data/all_data.csv')
# correlation = pd.DataFrame(df.drop(['start_of_week'],axis=1).corr()['total_boardings'])
# model_data = df[correlation.dropna().index.values].drop(['local_business_condition_index','high_school_off_season'], axis=1)
# features = model_data.drop(['total_boardings'], axis=1)
# features['runtime_hours'] = features['runtime_hours'].rolling(window=10,min_periods=1,center=True).mean()
# features = features.fillna(method='backfill').fillna(method='pad')

design = lhs(4, samples=10)