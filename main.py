import pandas as pd
import numpy as np
from features.extractor import extract_features
from dataset.dataset import surv_experiment, surv_ext_experiment, surv_ext_experiment_complete, office_test, hockey_test, movies_test
import tqdm
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# K = 8 
K = 12
# K = 16
W = 480
H = 360

def process_files(process_name, files, labels=[]):
    try:
        return pd.read_csv('{}.csv'.format(process_name)), []
    except Exception:
        file_features = []
        skip_ix = []
        tqdm_files = tqdm.tqdm(total=files.shape[0], desc=process_name, position=0)
        for i, f in enumerate(files):
            if len(labels) == len(files):
                features = extract_features(f, W, H, K, class_label=labels[i])
            else:
                features = extract_features(f, W, H, K)
            if len(features) > 0:
                file_features.append(features)
            else:
                skip_ix.append(i)
            tqdm_files.update(1)
        df = pd.DataFrame(file_features)
        df.to_csv('{}.csv'.format(process_name), index=False)
        print('{} Result Size: {}'.format(process_name, df.shape))
        return df, skip_ix

def get_lgbm_varimp(model, train_columns, max_vars=50):
    cv_varimp_df = pd.DataFrame([train_columns, model.feature_importances_]).T
    cv_varimp_df.columns = ['feature_name', 'varimp']
    cv_varimp_df.sort_values(by='varimp', ascending=False, inplace=True)    
    cv_varimp_df = cv_varimp_df.iloc[0:max_vars]
    return cv_varimp_df

def lgb_f1_score(y_true, y_pred):
    y_pred_ = np.round(y_pred) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_pred_), True

def main():
    # X_train, X_valid, y_train, y_valid = surv_experiment()

    # X_train, X_valid, y_train, y_valid = surv_ext_experiment()
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = surv_ext_experiment_complete()

    print('Processing train videos ...')
    train_df,_ = process_files('Train Features', X_train)#, y_train)

    print('Processing validation videos ...')
    valid_df, skip_ix = process_files('Validation Features', X_valid)
    if len(skip_ix) > 0:
        y_valid = np.delete(y_valid, skip_ix)
    
    print('Starting training...')
    # train
    
    print('-'*40)
    print('LGBM simple learn')
    print('-'*40)
    gbm = lgb.LGBMClassifier(num_leaves=40,
                            objective='binary',
                            learning_rate=0.05,
                            n_estimators=20)
    gbm.fit(train_df, y_train,
            eval_set=[(valid_df, y_valid)],
            # eval_metric = ['auc', 'binary_logloss'],
            eval_metric= lgb_f1_score,
            early_stopping_rounds=10)

    print(get_lgbm_varimp(gbm, train_df.columns))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(train_df.head(1))

    '''
    print()
    print('-'*40)
    print('LGBM cross-validation')
    print('-'*40)
    lgb_params = {
        'n_estimators': 20,
        'learning_rate': 0.01,
        'num_leaves': 40, 
        'objective': 'binary'}
    # form LightGBM datasets
    dtrain_lgb = lgb.Dataset(train_df, label=y_train)
    # LightGBM, cross-validation
    cv_result_lgb = lgb.cv(lgb_params, 
                        dtrain_lgb, 
                        nfold=4, 
                        stratified=True, 
                        early_stopping_rounds=10)
    best_params = gridsearch.best_params_
    print('Best parameters found by grid search are:', best_params)
    

    print()
    print('-'*40)
    print('LGBM grid search')
    print('-'*40)

    estimator = lgb.LGBMClassifier(learning_rate = 0.125,
                            objective='binary', 
                            n_estimators = 20, num_leaves = 38)

    param_grid = {
        'n_estimators': [x for x in range(24,40,2)],
        'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
    gridsearch = GridSearchCV(estimator, param_grid)

    gridsearch.fit(train_df, y_train,
            eval_set = [(valid_df, y_valid)],
            eval_metric = ['auc', 'binary_logloss'],
            early_stopping_rounds = 10)
    '''

    print()
    print('-'*40)
    print('Model Test')
    print('-'*40)

    
    # dataset which is not in train set
    # X_test, y_test = office_test(sample=True, ratio=2)
    
    # X_test, y_test = hockey_test()
    # X_test, y_test = movies_test()

    print('Processing test videos ...')
    test_df,_ = process_files('Test Features', X_test)
    y_pred = gbm.predict(test_df, num_iteration=gbm.best_iteration_)
    # y_pre  d = gridsearch.predict(test_df)
    print('The accuracy of prediction is:', accuracy_score(y_test[:y_pred.shape[0]], y_pred))
    print('The f1 of prediction is:', f1_score(y_test[:y_pred.shape[0]], y_pred))
    print('The roc_auc_score of prediction is:', roc_auc_score(y_test[:y_pred.shape[0]], y_pred))
    print('The null acccuracy is:', max(y_test.mean(), 1 - y_test.mean()))
    print('Confusion matrix')
    print(confusion_matrix(y_test[:y_pred.shape[0]], y_pred))

    fp1_check = X_test[(y_test == 1) & (y_pred == 0)]
    sorted(fp1_check)
    print(fp1_check)

    fp2_check = X_test[(y_test == 0) & (y_pred == 1)]
    sorted(fp2_check)
    print(fp2_check)

    # show results
    # shuffle
    x, _, y, _ = train_test_split(X_test, y_pred, test_size=0.00001, random_state=42)
    _,_ = process_files('View Features', x, y)


main()