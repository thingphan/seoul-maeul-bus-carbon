# 속도 모델 개발 노트북

AI 속도 모델([docs/SPEED_MODEL.md](../docs/SPEED_MODEL.md))을 설계하면서 수행한 비교 실험의 기록이다. 최종 모델은 이 실험 결과를 바탕으로 `src/models/`에 구현했으며, 공식 성능은 노트북 값이 아니라 `src/models/02_build_model_bc.py --mode validate`와 `src/transit/checks/validate_bc_tiers.py`의 검증 값이다.

| 노트북 | 질문 | 주요 결과 | 최종 설계에 반영된 내용 |
| --- | --- | --- | --- |
| [01_speed_model_baseline](01_speed_model_baseline.ipynb) | 도로 속성만으로 속도를 얼마나 맞히는가, 어떤 알고리즘이 나은가 | 원본 속성 23개·무작위 분할 기준 R²: 전체 평균 0.000, 제한속도 선형회귀 0.172, 도로등급 평균 0.266, LightGBM 0.610, XGBoost 0.632, CatBoost 0.556 | 트리 모델 채택, 단순 기준모형 대비 성능 비교 |
| [02_derived_features_lightgbm](02_derived_features_lightgbm.ipynb) | 어떤 파생·외부 변수가 처음 보는 도로의 속도 예측을 개선하는가 | 링크 단위 GroupKFold 기준 단계별 실험. 생활인구·도로망 위상·View-T·2칸 이웃·같은 도로명 구간 변수를 더해 R² 0.737(LightGBM)·0.740(XGBoost). 회전제한 정보(TURNINFO) 기반 변수는 C 계층을 악화시켜 제외 | 61개 변수 구성, XGBoost 채택. 단 0.740은 같은 시각 이웃 관측을 쓴 실험 상한이며, 실제 적용 모델은 과거 같은 시간대 이웃 평균을 써서 0.72 |
| [03_model_a_history](03_model_a_history.ipynb) | 관측 이력이 있는 도로는 무엇으로 예측하는가 | 시간 순서 분할(과거 기간 학습 → 이후 3일 평가)에서 이력 조회표 R² 0.925(MAE 2.32km/h), 과거 패턴 변수를 더한 LightGBM 0.928. 요일별 7단계 조회표는 과적합 | 학습 없는 이력 조회표(Model A), 평일/주말 2단계 |
| [04_tier_split_evaluation](04_tier_split_evaluation.ipynb) | 이웃속도 변수는 어떤 도로에서 통하고 어디서 깨지는가 | 미관측 도로를 이웃 관측 유무(B/C)로 나눠 평가. 이웃속도는 B에서 크게 도움이 되지만, 이웃이 없는 C에서는 결측 변수를 공유한 단일 모델의 성능이 떨어짐 | B·C 분리 모델, 계층별 성능 보고 |

## 읽는 법

- 노트북 안의 '○차'는 실험 순서 번호이며, 각 단계의 변수 추가·제거와 그 효과를 기록했다.
- 무작위 분할은 같은 도로가 학습과 평가에 섞여 점수가 후하게 나오므로, 최종 비교는 도로 단위 분할(GroupKFold)로 했다.
- 출력 셀은 실행 당시 결과를 그대로 보존했다.

## 재실행

02는 현재 파이프라인 산출물(`seoul_link_dimension.csv`, `seoul_link_speed_fact.csv`)을 읽는다. 01·03·04는 모델 개발 초기의 중간 파일(예: 결합 전 링크·속도 표, 당시 전 링크 예측 파일명)을 읽도록 작성되어 있어 현재 파이프라인 산출물만으로는 그대로 다시 실행되지 않는다. 이 노트북들의 결론을 반영한 최종 모델은 `src/models/`에서 재현된다.
