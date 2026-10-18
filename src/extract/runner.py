"""LLM 추출 공통 실행기: 병렬 배치 호출 + 결과를 jsonl에 이어 쓰기.

중간에 끊겨도 다시 돌리면 이미 처리한 id는 건너뛴다.
"""
import json
from pathlib import Path
from typing import Callable, Iterable

from tqdm import tqdm


def done_ids(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)[key])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def run_batches(chain, jobs: list[dict], to_records: Callable[[dict, object], Iterable[dict]],
                out_path: Path, batch_size: int = 32, max_concurrency: int = 8, unit: str = "건"):
    """jobs: [{"input": <chain 입력>, ...메타}], to_records(job, 결과) → 저장할 레코드들.
    하나 끝날 때마다 바로 파일에 쓰고 진행 표시를 올린다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0
    with open(out_path, "a", encoding="utf-8") as f, \
            tqdm(total=len(jobs), desc=out_path.stem, unit=unit) as bar:
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            for idx, res in chain.batch_as_completed([j["input"] for j in chunk],
                                                     config={"max_concurrency": max_concurrency},
                                                     return_exceptions=True):
                if isinstance(res, Exception) or res is None:
                    n_fail += 1
                    if n_fail <= 3:  # 처음 몇 개는 원인을 보여준다
                        tqdm.write(f"  실패: {type(res).__name__}: {str(res)[:200]}")
                else:
                    for rec in to_records(chunk[idx], res):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                bar.update(1)
                bar.set_postfix(성공=n_ok, 실패=n_fail)
    print(f"[{out_path.name}] 성공 {n_ok}, 실패 {n_fail} (실패분은 다시 실행하면 재시도)")
