"""AI 속도 모델 B·C의 등급별 검증 성능(R²·MAE·RMSE·오차 분위수)을 기록한다.

src/models/02_build_model_bc.py --mode validate(도로 단위 3-fold GroupKFold)를 그대로 실행하면서,
폴드마다 B분리·C분리·C단일 순으로 호출되는 eval_bucket의 결과를 가로채 등급별로 모은다. 모델 코드는 바꾸지 않는다.

출력: data/processed/model_bc_tier_validation.json (폴드별 값과 표본 수 가중 평균)
사용: python src/transit/checks/validate_bc_tiers.py   (전체 검증을 다시 돌리므로 수십 분 걸린다)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BC_SCRIPT = PROJECT_ROOT / "src" / "models" / "02_build_model_bc.py"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "model_bc_tier_validation.json"
LABELS = ["B", "C", "C_single"]
METRICS = ("mae", "rmse", "r2", "abs_err_p50", "abs_err_p80", "abs_err_p90")


def main() -> None:
    spec = importlib.util.spec_from_file_location("model_bc", BC_SCRIPT)
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)

    records: list[dict] = []
    original = bc.eval_bucket

    def recording_eval_bucket(y_true, pred, mask):
        result = original(y_true, pred, mask)
        err = np.abs(np.asarray(y_true)[mask] - np.asarray(pred)[mask])
        extra = {f"abs_err_p{q}": float(np.percentile(err, q)) for q in (50, 80, 90)} if mask.sum() else {}
        records.append({"fold": len(records) // 3, "tier": LABELS[len(records) % 3],
                        **{k: float(v) for k, v in result.items()}, **extra})
        return result

    bc.eval_bucket = recording_eval_bucket
    start = time.time()
    sys.argv = ["02_build_model_bc.py", "--mode", "validate"]
    bc.main()

    summary = {}
    for tier in LABELS:
        rows = [r for r in records if r["tier"] == tier]
        n = np.array([r["n"] for r in rows], dtype=float)
        summary[tier] = {k: float(np.average([r[k] for r in rows], weights=n)) for k in METRICS}
        summary[tier]["n_total"] = int(n.sum())
    OUT_PATH.write_text(json.dumps({"folds": records, "weighted_mean": summary, "minutes": (time.time() - start) / 60},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
