
from sklearn.model_selection import cross_validate
from pandas import DataFrame
from pandas import MultiIndex


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

    if not return_estimator:
        return scores_df
    else:
        estimators =  scores['estimator']
        return scores_df, estimators
