import itertools

import arff
import pandas as pd

from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    redirect,
    request,
)
from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)
from scipy.io import arff


load_dotenv()
app = Flask(__name__)

data, meta = arff.loadarff(open('datasets/supermarket.arff', 'r'))
df = pd.DataFrame(data).drop('total', axis=1)

map_f = lambda v: 0 if (v == b'?' or v == b'low') else 1
df = df.applymap(map_f)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
rules = association_rules(
    frequent_itemsets, metric='confidence', min_threshold=0.2)

itemset_count = len(frequent_itemsets)
rules_count = len(rules)

items = sorted(
    set(itertools.chain.from_iterable(frequent_itemsets.itemsets.values)))
basket = set()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form_items = request.form.getlist('items')
        basket.update(form_items)

    recommendations = set()
    if basket:
        frozen_basket = frozenset(basket)
        rules_filtered = rules[rules['antecedants'] == frozen_basket]
        consequents = rules_filtered.tail(8)['consequents'].values

        recommendations.update(
            set(itertools.chain.from_iterable(consequents))
        )

    context = {
        'itemset_count': itemset_count,
        'rules_count': rules_count,
        'items': items,
        'basket': basket,
        'recommendations': recommendations,
    }
    return render_template('index.html', **context)


@app.route('/reset-basket/', methods=['POST'])
def reset_basket():
    global basket
    basket = set()
    return redirect('/')
