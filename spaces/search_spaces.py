# search space for hyperparameter optimization

from hyperopt import hp
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb


xgb_space = {'model': xgb.XGBClassifier,
             'params': {
                        'n_estimators' : hp.quniform('xgb_n', 1000, 5000, 500),
                        'learning_rate' : hp.normal('xgb_eta', 0.05, 0.01),
                        'max_depth' : hp.quniform('xgb_max_depth', 2, 8, 1),
                        'min_child_weight' : hp.quniform('xgb_min_child_weight', 1, 6, 1),
                        'subsample' : hp.uniform('xgb_subsample', 0.2, 1),
                        'gamma' : hp.uniform('xgb_gamma', 0.01, 0.4),
                        'colsample_bytree' : hp.uniform('xgb_colsample_bytree', 0.2, 1),
                        'objective': hp.choice('xgb_obj', ['binary:logistic']),
                        'scale_pos_weight': hp.uniform('xgb_w', 1.0, 5.0)
                        }
            }

lgb_space = {'model': lgb.LGBMClassifier,
             'params': {
                      'class_weight': hp.choice('class_weight', [None, 'balanced']),
                      'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                                                   {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                                   {'boosting_type': 'goss', 'subsample': 1.0}]),
                      'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
                      'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                      'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
                      'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
                      'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                      'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                      'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
                        }
            }

rf_space = {'model': RandomForestClassifier,
            'params': {
                      'max_depth': hp.choice('max_depth', range(1,20)),
                      'max_features': hp.choice('max_features', range(1,150)),
                      'n_estimators': hp.choice('n_estimators', range(100,500)),
                      'criterion': hp.choice('criterion', ["gini", "entropy"]),
		      'class_weights': 'balanced'
                      }
            }


gbc_space = {'model': GradientBoostingClassifier,
			 'params': {
					'learning_rate': hp.uniform( 'lr', 0.01, 0.2 ),
					'subsample': hp.uniform( 'ss', 0.8, 1.0 ),
					'max_depth': hp.quniform( 'md', 2, 10, 1 ),
					'max_features': hp.choice( 'mf', ( 'sqrt', 'log2', None )),
					'min_samples_leaf': hp.quniform( 'msl', 1, 10, 1 ),
					'min_samples_split': hp.quniform( 'mss', 2, 20, 1 )
					}
			}