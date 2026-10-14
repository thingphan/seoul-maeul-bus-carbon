"""경유·CNG 중형 마을버스의 1km당 CO2를 전기버스 모델값에서 환산한다.

VT-CPFM 버스 모델(Edwardes & Rakha 2014, TRR 2428; Wang 2016 버지니아공대 박사논문 표 3.9)은
초 단위 속도·가속도를 입력으로 보정된 계수라, 이 프로젝트의 링크 내 정속(가속도 0) 근사에
그대로 넣으면 정류장 정지·출발 연료가 빠져 버스가 과소 추정된다. 그래서 같은 서울 버스 주행
패턴에서 전기버스와 경유·CNG 버스를 함께 측정한 실측 비율로 환산한다.

    근거: 서울의 CNG버스를 전기버스로 대체했을 때 환경 개선 효과 분석(전기전자학회논문지 24(3),
          2020, DOI 10.7471/ikeee.2020.24.3.821)
          서울 시내버스 평균 주행속도(19.8km/h) 패턴에서
            전기버스 0.7km/kWh(=1.43kWh/km), CNG 721gCO2/km, 경유 723gCO2/km

    경유버스 gCO2/km = 전기버스 kWh/km(04, 노선별 속도·경사 반영) × 723 / (1/0.7)
    CNG버스  gCO2/km = 전기버스 kWh/km × 721 / (1/0.7)

노선 간 차이(정체·언덕)는 전기버스 물리모델이, 연료 간 차이는 서울 실측 비율이 담당한다.
마을버스 크기는 중형(현대 카운티)으로 고정한다.
"""

from __future__ import annotations

SEOUL_EBUS_KWH_PER_KM = 1.0 / 0.7
SEOUL_DIESEL_G_PER_KM = 723.0
SEOUL_CNG_G_PER_KM = 721.0

DIESEL_G_PER_EBUS_KWH = SEOUL_DIESEL_G_PER_KM / SEOUL_EBUS_KWH_PER_KM
CNG_G_PER_EBUS_KWH = SEOUL_CNG_G_PER_KM / SEOUL_EBUS_KWH_PER_KM


def diesel_g_per_km(ebus_kwh_per_km):
    return ebus_kwh_per_km * DIESEL_G_PER_EBUS_KWH


def cng_g_per_km(ebus_kwh_per_km):
    return ebus_kwh_per_km * CNG_G_PER_EBUS_KWH
