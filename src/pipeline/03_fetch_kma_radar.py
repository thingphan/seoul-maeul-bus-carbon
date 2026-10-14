"""KMA HSR 기반 격자 위경도 및 1시간 누적 레이더 강수량을 내려받는다.

구 src/09_fetch_kma_radar.py와 로직 동일, src/pipeline/으로 위치만 이동.

출력: data/raw/weather/grid/*.txt, data/raw/weather/radar/*.txt

사용:
  python src/pipeline/03_fetch_kma_radar.py --grid
  python src/pipeline/03_fetch_kma_radar.py --from-5min-times
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
WEATHER_RAW_DIR = RAW_DATA_DIR / "weather"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
KMA_API_KEY_PATH = WEATHER_RAW_DIR / "kma_api_key.txt"
FIVEMIN_SPEED_PATH = INTERIM_DATA_DIR / "seoul_link_speed_5min_hourly.csv"

GRID_DIR = WEATHER_RAW_DIR / "grid"
RADAR_DIR = WEATHER_RAW_DIR / "radar"

LATLON_API_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-rdr_latlon_api"
RADAR_API_URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-rdr_cmp1_api"


def load_api_key() -> str:
    env_key = os.environ.get("KMA_API_KEY", "").strip()
    if env_key:
        return env_key
    if KMA_API_KEY_PATH.exists():
        file_key = KMA_API_KEY_PATH.read_text(encoding="utf-8").strip()
        if file_key:
            return file_key
    raise SystemExit(
        f"KMA API 키가 없습니다. {KMA_API_KEY_PATH} 파일을 만들거나 "
        "환경변수 KMA_API_KEY를 설정하세요."
    )


def validate_tm(value: str) -> str:
    if len(value) != 12 or not value.isdigit():
        raise SystemExit(f"tm은 YYYYMMDDHHMM 형식의 12자리 숫자여야 합니다: {value}")
    try:
        datetime.strptime(value, "%Y%m%d%H%M")
    except ValueError as exc:
        raise SystemExit(f"실제 날짜/시간이 아닙니다: {value}") from exc
    return value


def iter_tms(start_tm: str, end_tm: str, step_minutes: int):
    start = datetime.strptime(validate_tm(start_tm), "%Y%m%d%H%M")
    end = datetime.strptime(validate_tm(end_tm), "%Y%m%d%H%M")
    if start > end:
        raise SystemExit("start-tm이 end-tm보다 늦습니다.")
    if step_minutes < 1:
        raise SystemExit("step-minutes는 1 이상이어야 합니다.")
    current = start
    while current <= end:
        yield current.strftime("%Y%m%d%H%M")
        current += timedelta(minutes=step_minutes)


def build_url(base_url: str, params: dict[str, str]) -> str:
    return f"{base_url}?{urlencode(params)}"


def masked_url(url: str, api_key: str) -> str:
    return url.replace(api_key, "***KMA_API_KEY***")


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "seoul-traffic-analysis/0.1"})
    try:
        with urlopen(request, timeout=180) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset, errors="replace")
    except HTTPError as exc:
        charset = exc.headers.get_content_charset() or "utf-8"
        body = exc.read().decode(charset, errors="replace")
        raise SystemExit(f"HTTP 오류 {exc.code}: {exc.reason}\n{body[:1000]}") from exc
    except URLError as exc:
        raise SystemExit(f"URL 요청 오류: {exc}") from exc


def save_text(url: str, out_path: Path, api_key: str, overwrite: bool = False) -> None:
    if out_path.exists() and not overwrite:
        print("이미 있음:", out_path)
        return
    print("요청:", masked_url(url, api_key))
    text = fetch_text(url)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print("저장:", out_path)
    print("응답 앞부분:")
    print(text[:500])


def fetch_grid(api_key: str, overwrite: bool = False) -> None:
    for latlon in ["lon", "lat"]:
        url = build_url(LATLON_API_URL, {"cmp": "HSR", "latlon": latlon, "disp": "A", "authKey": api_key})
        save_text(url, GRID_DIR / f"hsr_{latlon}.txt", api_key, overwrite=overwrite)


def fetch_radar(api_key: str, tm: str, overwrite: bool = False) -> None:
    url = build_url(
        RADAR_API_URL,
        {"tm": validate_tm(tm), "cmp": "PCPH", "qcd": "MSK", "obs": "ECHO", "acc": "60", "map": "HB", "disp": "A", "authKey": api_key},
    )
    save_text(url, RADAR_DIR / f"hsr_pcph_acc60_{tm}.txt", api_key, overwrite=overwrite)


def speed_row_to_tm(date_value, time_code_value) -> str:
    date = datetime.strptime(str(int(date_value)), "%Y%m%d")
    time_code = int(time_code_value)
    if time_code == 24:
        return (date + timedelta(days=1)).strftime("%Y%m%d") + "0000"
    return date.strftime("%Y%m%d") + f"{time_code:02d}00"


def iter_5min_speed_tms():
    if not FIVEMIN_SPEED_PATH.exists():
        raise SystemExit(
            f"5분 집계 속도 파일이 없습니다: {FIVEMIN_SPEED_PATH}\n"
            "src/pipeline/02_aggregate_5min_speed.py를 먼저 실행하세요."
        )
    times = pd.read_csv(FIVEMIN_SPEED_PATH, usecols=["기준일", "시간코드"], encoding="utf-8-sig").drop_duplicates()
    tms = {speed_row_to_tm(d, t) for d, t in zip(times["기준일"], times["시간코드"])}
    yield from sorted(tms)


def main() -> None:
    parser = argparse.ArgumentParser(description="KMA HSR 기반 1시간 누적 레이더 강수량 데이터를 내려받습니다.")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--tm")
    parser.add_argument("--start-tm")
    parser.add_argument("--end-tm")
    parser.add_argument("--step-minutes", type=int, default=60)
    parser.add_argument("--from-5min-times", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.grid and not args.tm and not (args.start_tm and args.end_tm) and not args.from_5min_times:
        parser.error("--grid, --tm, --start-tm/--end-tm, 또는 --from-5min-times 중 하나가 필요합니다.")

    api_key = load_api_key()

    if args.grid:
        fetch_grid(api_key, overwrite=args.overwrite)
    if args.tm:
        fetch_radar(api_key, args.tm, overwrite=args.overwrite)
    if args.start_tm and args.end_tm:
        for tm in iter_tms(args.start_tm, args.end_tm, args.step_minutes):
            fetch_radar(api_key, tm, overwrite=args.overwrite)
    if args.from_5min_times:
        failed_tms = []
        for tm in iter_5min_speed_tms():
            try:
                fetch_radar(api_key, tm, overwrite=args.overwrite)
            except (Exception, SystemExit) as exc:
                print(f"[{tm}] 다운로드 실패:", exc)
                failed_tms.append(tm)
        if failed_tms:
            failed_path = WEATHER_RAW_DIR / "radar_failed_tms.txt"
            failed_path.write_text("\n".join(failed_tms), encoding="utf-8")
            print("실패 시각 저장:", failed_path)


if __name__ == "__main__":
    main()
