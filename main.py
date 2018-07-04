import arff
import pandas as pd

from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)
from scipy.io import arff

data, meta = arff.loadarff(open('datasets/supermarket.arff', 'r'))
df = pd.DataFrame(data)

map_f = lambda v: 0 if (v == b'?' or v == b'low') else 1
df = df.applymap(map_f)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
rules = association_rules(
    frequent_itemsets, metric='confidence', min_threshold=1)
