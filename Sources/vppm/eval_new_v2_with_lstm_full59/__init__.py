"""[new_v2] AMMTO Spatial Variation Baseline 빌드에 학습된 LSTM_FULL59 모델을 part-level 로 적용하는 평가 전용 패키지.

학습은 안 함. 신규 빌드 1개 (`2023-03-15 AMMTO Spatial Variation Baseline.hdf5`) 에 대해
21-feat 추출 + 6 시퀀스 캐시 빌드 → 학습된 5-fold 모델 ensemble inference → part 단위
평균 → parts/test_results (YS/UTS/TE) 와 비교. UE 는 GT 부재로 prediction 만 저장.
"""
