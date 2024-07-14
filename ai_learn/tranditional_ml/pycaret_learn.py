# load sample dataset
import pandas as pd
from pycaret.datasets import get_data

# pd.set_option('display.max_columns', None)
data = get_data('diabetes')

from pycaret.classification import *
s = setup(data, target = 'Class variable', session_id = 123)
print(s)

# functional API
best = compare_models()

# functional API
evaluate_model(best)