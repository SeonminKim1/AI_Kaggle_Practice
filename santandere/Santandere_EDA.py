# 산탄데르 제품 추천대회

# 1-1. Training Data 읽어오기
import pandas as pd
import numpy as np
data_path = 'D:/dataset/santander-product-recommendation/train_ver2/train_ver2.csv'
trn=pd.read_csv(data_path)
print('데이터 로드 완료')

# 1-2. Training Data 데이터 미리보기
# 현재 pandas의 Dataframe 형태임
print('** Data Shape\n', trn.shape, '\n') # 데이터의 형태

print('** Data Columns name')
for cols in range(0, len(trn.columns)):
    print('{}'.format(trn.columns[cols]), end=', ')

# pandas는 info로 정보를 제공함.
print('\n** Data info of columns')
# print(trn.info)

# 1-3 데이터 5줄만 보기
trn.head()

"""
### 변수 설명
- fecha_dato : 날짜기록 / ncodpers: 고객 고유 식별번호 
- ind_empleado(고용지표 A:active, B:ex employed, F:filial, N:not employee, P:passive)
- pais_residencia: 고객거주국가 / sexo: 성별
- fecha_alta: 고객이 은행과 첫 계약 체결 날짜 / ind_nuevo: 신규 고객지표 (6개월 이내 신규고객인 경우 값 1)
- antiguedada(은행 거래 누적 기간) / indrel: 고객등급(1-1등급고객, 99:해당 달에 고객 1등급이 해제되는 1등급 고객) 
- ult_fec_cli_1t: 1등급 고객으로서 마지막 날짜 
- indrel_1mes: 월초기준고객등급(1: 1등급, 2:co-owner, P:potential, 3:former primary, 4:former-co-owner)
- tiprel_1mes: 월초기준 고객관계유형(A:active, I:inactive, P:former customer, R:potential)
- indresi: 거주지표(고객의 거주 국가와 은행이 위치한 국가 동일 여부:S-yes, N(no))
- indext: 외국인지표(고객의 태어난 국가와 은행이 위치한 국가 동일 여부(S,N))
- conyuemp:배우자지표(1:은행지원을 배우자로 둔 고객) / canal_entrada(고객유입채널)
- indfall:고객 사망여부 / tipodom : 주소유형(1:primary address)  
- cod_prov(지방코드) / nomprov(지방이름)
- ind_actividad_cliente(활발성 지표(1:active customer, 2:inactive customer))
- renta:가구총수입 / segmento:분류 (01:VIP, 02:개인, 03:대출)

- ind_ahor_fin_ult1 (예금) / ind_aval_fin_ult1(보증) / ind_cco_fin_ult1(당좌 예금) 
- ind_cder_fin_ult1(파생 상품 계좌) / ind_cno_fin_ult1(급여 계정)
- ind_ctju_fin_ult1 (청소년 계정) / ind_ctma_fin_ult1(마스 특별 계정)
- ind_ctop_fin_ult1(특정 계정) / ind_ctpp_fin_ult1(특정 플러스 계정)
- ind_deco_fin_ult1 (단기 예금) / ind_deme_fin_ult1(중기 예금) 
- ind_delav(장기 예금) / ind_ecue_fin_ult1(e-계정) / ind_fond_fin_ult1(펀드)
- ind_hip_fin_ult1 (부동산 대출) / ind_plan_fin_ult1(연금) 
- ind_pres_fin_ult1(대출) / ind_reca_fin_ult1 (세금) / ind_tjcr_fin_ult1(신용카드)
- ind_valo_fin_ult1 (증권) / ind_viv_fin_ult1 (홈계정) / ind_nomina_ult1(급여)
- ind_nom_pnes_ult1(연금) / ind_recibo_ult1(직불 카드)
"""

# 1-4. 24개의 주요 변수 중 수치형 변수 살펴보기 (7개)
# print(trn['ncodpers']) # 'ncodpers' 열에 대한 정보 출력
num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64','float64']]
trn[num_cols].describe()

"""
### 수치형변수 분석결과
- ncodpers : 최솟값 15889, 최댓값 1553689를 갖는 고유식별번호
- ind_nuevo : 최소 75%값이 0이며, 나머지가 값 1을 가지는 신규 고객 지표
- indrel : 최소 75%값이 1이며, 나머지가 값 99를 갖는 고객 등급변수
- tipodom : 모든값이 1인 주소 유형 변수 (학습에 도움이 안될 것)
- cod_prov : 최소 1에서 최대 52의 값을 가지며 수치형이지만 범주형으로써의 의미를 갖는 지방 코드 변수
- ind_activated_cliente : 최소 50%값이 0이며, 나머지가 값 1을 가지는 활발성 지표
- renta :최소 1203.73 ~ 최대 2889440의 값을 가지는 전형적인 수치형 변수 가구 총 수입을 나타냄.
"""

# 1-5. 범주형 변수 살펴보기 (17개) / 13619575행
# 범주형 변수의 pandas.describe()는 좀 다른 행의미가 나옴.
cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']] # 0은 object라는 뜻
trn[cat_cols].describe()

"""
### 범주형변수 분석결과
- count : 해당 변수들 유효한 데이터갯수를 의미 (나머지는 결측값) , 13619575개가 정상
- unique : 해당 범주형 변수의 고윳값 갯수 (sexo의 경우 2개임)
- top : 가장 빈도가 높은 데이터
- freq : top에서 표시된 최빈데이터의 빈도수
"""