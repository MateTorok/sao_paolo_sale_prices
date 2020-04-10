
from sklearn.model_selection import cross_validate
from pandas import DataFrame
from pandas import MultiIndex
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from numpy import round

def get_cros_val_scores(model, X , y, 
                        scoring = ['neg_mean_absolute_error', 'neg_root_mean_squared_error', 'r2'],
                        cv=10,
                        return_train_score = True,
                        return_estimator = False):
    '''Returns a dataframe with the scores (without time),
    if return_estimator then returns a tuple (scores, estimators)'''
    scores = cross_validate(model, X, y, 
                            cv=cv,
                            scoring= scoring,
                            return_train_score=return_train_score,
                            return_estimator=return_estimator)
    
    score_names = ['test_' + s for s in scoring]
    if return_train_score:
        train_scores = ['train_' + s for s in scoring]
        score_names = train_scores + score_names
        index = MultiIndex.from_tuples( [tuple(s.split('_', maxsplit=1)) for s in score_names],
                                        names = ['set', 'score'] )
    else:
        index = scoring

    only_scores = {item: scores[item] for item in score_names }
    scores_df = DataFrame([*only_scores.values()],  index = index).transpose()

    for col in scores_df.columns:
        if 'neg_' in col:
            scores_df.rename(columns={col:col[4:]}, inplace=True)
            scores_df[col] = scores_df[col].abs()

    if not return_estimator:
        return scores_df
    else:
        estimators =  scores['estimator']
        return scores_df, estimators

def mean_scores_format(scores):
    scores.rename({'neg_mean_absolute_error': 'MAE',
                    'neg_root_mean_squared_error': 'RMSE'}, 
                    level=1, axis =1, inplace = True)
    return round(scores.agg(['mean', 'std']).stack().abs(),3).style.format('{0:,}')


def get_metrics(y_act, y_pred, return_dict=True):
    scores = {'MAE': mean_absolute_error(y_act, y_pred),
            'RMSE': mean_squared_error(y_act, y_pred)**0.5,    
            'R2': r2_score(y_act, y_pred) }
    if return_dict:
        return scores
    else:
        print('\n', round( DataFrame(scores, index=['scores']),3 ))


def get_scores(model, X, y, exp=False):
    model.fit(X,y)
    y_pred = model.predict(X)
    if exp:
        y = np.exp(y) 
        y_pred = np.exp(y_pred)
    scores = ev.get_metrics(y, y_pred)
    return scores