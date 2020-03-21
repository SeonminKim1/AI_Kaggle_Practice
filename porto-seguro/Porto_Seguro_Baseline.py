"""
### porto-seguro Baseline 모델
- 탐색적 데이터 분석 과정을 통해 확인결과 변수들은 모두 익명화되어있고 값들은 숫자로 치환되어있음.
- 범주형 변수도 이미 숫자로 치환되어있어, 데이터 전처리를 수행할 필요가 없을 만큼 데이터가 깔끔하고 깨끗함.
"""

import pandas as pd
# 2-1. 훈련 데이터, 테스트 데이터 읽어오기.
data_path = 'D:/dataset/porto-seguro-safe-driver-prediction/'
trn = pd.read_csv(data_path + 'train.csv')
tst = pd.read_csv(data_path + 'test.csv')

"""
##### 실제 변수 이름을 알수 없어 피쳐 엔지니이렁시 초기에 방향성 찾기가 어려울 수 있음.
##### => 3가지 기초 피쳐 엔지니어링 수행(파생변수 생성)
- (1) 결측값의 갯수를 나타내는 missing 변수 
- (2) 이진 변수들의 총합
- (3) Target Encoding 파생 변수
"""

# 실제 변수 이름을 알수 없어 피쳐 엔지니이렁시 초기에 방향성 찾기가 어려울 수 있음.
# 3가지 기초 피쳐 엔지니어링 수행(파생변수 생성)
# 2-2.  3가지 기초 피처 엔지니어링 수행 (파생 변수 생성)

train_label = trn['target']
train_id = trn['id']
test_id = tst['id']
del trn['target']
del trn['id'] # id 와 target 삭제
del tst['id'] # id삭제

# 파생변수 01 : 결측값을 의미하는 '-1' 의 갯수를 센다.
# 결측값의 합을 파생변수로 사용하는 이유는 해당 파생변수는 손쉽게 만들 수 있으며 효자 변수로 작용한 경우가 종종 있음.
# 해당 대회 예시로는 결측값의 갯수가 데이터 내에 새로운 군집정보를 제공가능
# ex1) 초보 운전자들의 정보가 적을 수 있음 / ex2) 전국의 지부에서 모았을 때 특정 지점의 수집 오류 존재해서 결측값 처리 됬었을 수도 있음.
trn['missing'] = (trn==-1).sum(axis=1).astype(float) # 가로축에 대한 missing 값을 세서 float형으로 나타냄
tst['missing'] = (tst==-1).sum(axis=1).astype(float) #
print(trn['missing'])
print(tst['missing'])

# 파생 변수 02 : 이진 변수의 합
# 이진 변수는 값이 0 혹은 1이기 때문에 각 변수가 파생 변수에 미치는 영향력이 균등
# 실수나 범주형 변수 간의 상호 작용 변수를 생성시 변수별 영향력 조절하는 작업 필요
bin_features = [c for c in trn.columns if 'bin' in c]
trn['bin_sum'] = trn[bin_features].sum(axis=1)
tst['bin_sum'] = tst[bin_features].sum(axis=1)
print(trn['bin_sum'])
print(tst['bin_sum'])

# 파생 변수 03 : 데이터 탐색 분석 과정에서 선별한 일부 변수를 대상으로 Target Encoding을 수행한다. Target Encoding은 교차 검증 과정에서 진행한다.
# Target Encoding은 단일변수의 고유값별 타겟 변수의 평균값을 파생 변수로 활용하는 피쳐 엔지니어링 기법. 주로 범주형에서 좋은 성능을 보임.

features = ['ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_12_bin', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_car_01_cat',
            'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_11_cat', 'ps_ind_01', 'ps_ind_03', 'ps_ind_15', 'ps_car_11']

# 2-3 LightGBM 모델의 설정 및 Gini 함수 정의.
num_boost_round = 100 #10000
params = {"objective": "binary",
          "boosting_type": "gbdt",
          "learning_rate": 0.1,
          "num_leaves": 15,
          "max_bin": 256,
          "feature_fraction": 0.6,
          "verbosity": 0,
          "drop_rate": 0.1,
          "is_unbalance": False,
          "max_drop": 50,
          "min_child_samples": 10,
          "min_child_weight": 150,
          "min_split_gain": 0,
          "subsample": 0.9,
          "seed": 2018
}
print('LightGBM 모델 정의 완료')

# Gini 함수 정의
import numpy as np

def Gini(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]

    # sort rows on prediction column
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:, 0].argsort()][::-1, 0]
    pred_order = arr[arr[:, 1].argsort()][::-1, 0]

    # get Lorenz curves
    L_true = np.cumsum(true_order) * 1. / np.sum(true_order)
    L_pred = np.cumsum(pred_order) * 1. / np.sum(pred_order)
    L_ones = np.linspace(1 / n_samples, 1, n_samples)

    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)

    # normalize to true Gini coefficient
    return G_pred * 1. / G_true

cv_only = True
save_cv = True
full_train = False

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', Gini(labels, preds), True

print('Gini함수 정의 완료')

# 2-4. Stratified 5-Fold 내부 교차 검증을 준비
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgbm

# 원래는 5 FOLD 인데 시간상 2FOLD 만 반복
NFOLDS = 2  # 5

# 분리된 데이터 폴드내의 타겟 변수의 비율을 유지하기 위해 사이킷런의 StratifiedKFold 함수 사용.
# 재현성을 위해 random_state는 고정
kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=218)
kf = kfold.split(trn, train_label)

cv_train = np.zeros(len(train_label))
cv_pred = np.zeros(len(test_id))
best_trees = []
fold_scores = []

# 2-5.  5-FOLD이므로 5번 반복
for i, (train_fold, validate) in enumerate(kf):
    # 훈련/검증 데이터를 분리한다
    X_train, X_validate, label_train, label_validate = trn.iloc[train_fold, :], trn.iloc[validate, :], train_label[
        train_fold], train_label[validate]

    # target encoding 피쳐 엔지니어링을 수행한다
    for feature in features:
        # 훈련 데이터에서 feature 고유값별 타겟 변수의 평균을 구한다
        map_dic = pd.DataFrame([X_train[feature], label_train]).T.groupby(feature).agg('mean')
        map_dic = map_dic.to_dict()['target']
        # 훈련/검증/테스트 데이터에 평균값을 매핑한다
        X_train[feature + '_target_enc'] = X_train[feature].apply(lambda x: map_dic.get(x, 0))
        X_validate[feature + '_target_enc'] = X_validate[feature].apply(lambda x: map_dic.get(x, 0))
        tst[feature + '_target_enc'] = tst[feature].apply(lambda x: map_dic.get(x, 0))
        print(feature, 'ok ', end=',')

    dtrain = lgbm.Dataset(X_train, label_train)
    dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)

    # 훈련 데이터를 학습하고, evalerror() 함수를 통해 검증 데이터에 대한 정규화 Gini 계수 점수를 기준으로 최적의 트리 개수를 찾는다.
    bst = lgbm.train(params, dtrain, num_boost_round, valid_sets=dvalid, feval=evalerror, verbose_eval=100,
                     early_stopping_rounds=100)  # early_stopping_rounds=1000
    best_trees.append(bst.best_iteration)

    # 테스트 데이터에 대한 예측값을 cv_pred에 더한다.
    cv_pred += bst.predict(tst, num_iteration=bst.best_iteration)
    cv_train[validate] += bst.predict(X_validate)

    # 검증 데이터에 대한 평가 점수를 출력한다.
    score = Gini(label_validate, cv_train[validate])
    print('\n', i, '번째 score:', score, ' \n\n')
    fold_scores.append(score)

# k번 학습한 모델의 예측값을 평균냄
cv_pred /= NFOLDS # NFOLDS 갯수로 나눔

# 시드값별로 교차 검증 점수를 출력한다.
print('  GINI 계수 값 : ', Gini(train_label, cv_train))
print('Fold 실행 점수 :', fold_scores)
print(best_trees, np.mean(best_trees))

# 테스트 데이터에 대한 결과물을 저장한다.
pd.DataFrame({'id': test_id, 'target': cv_pred}).to_csv('lgbm_baseline.csv', index=False)

"""
##### 결론 요약
- (1) 주최측에서 데이터 익명화와 전처리 과정을 꼼꼼하게 진행하여 별도로 데이터 정제 및 전처리를 수행 할 필요가없다
- (2) 3가지 파생변수 생성시 결측값의 갯수를 나타내는 'missing변수' / 이진 변수들의 합을 나타내는 상호 작용 변수 / 데이터 탐색분석과정에서 얻은 예측력이 높은 일부 변수들을 대상으로 진행한 Target Encoding 파생변수 생성
- (3) 2번의 Fold (원래는 5번)에서 학습한 모델의 예측값을 평균하여 테스트데이터에 대한 최종 예측값 산출
"""


