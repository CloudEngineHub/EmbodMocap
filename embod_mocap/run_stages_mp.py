import argparse
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_PATH = os.path.join(SCRIPT_DIR, "run_stages.py")


@dataclass
class Task:
    task_id: str
    kind: str  # scene | seq
    scene_folder_rel: str
    scene_folder_abs: str
    seq_name: str
    steps: List[int]
    xlsx_path: str
    retry: int = 0


def parse_steps(steps_raw: str) -> List[int]:
    arg_steps = steps_raw.split(',')
    steps: List[int] = []
    for s in arg_steps:
        s = s.strip()
        if not s:
            continue
        if s.isdigit():
            steps.append(int(s))
        elif '-' in s:
            start, end = s.split('-')
            steps.extend(range(int(start), int(end) + 1))
        else:
            raise ValueError(f"Invalid stage: {s}")

    seen = set()
    steps_uniq = []
    for st in steps:
        if st not in seen:
            seen.add(st)
            steps_uniq.append(st)
    return steps_uniq


def parse_gpu_ids(gpu_ids_raw: str) -> List[int]:
    if gpu_ids_raw is None or gpu_ids_raw.strip() == "":
        return [0]
    out = []
    for part in gpu_ids_raw.split(','):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else [0]


def load_xlsx_rows(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    if len(xl.sheet_names) == 1:
        return pd.read_excel(xlsx_path)
    dfs = [pd.read_excel(xlsx_path, sheet_name=sn) for sn in xl.sheet_names]
    return pd.concat(dfs, ignore_index=True)


def get_bool_from_excel(row, column, default=False):
    value = row.get(column, default)
    if value is None or pd.isna(value):
        return default
    return str(value).upper() in ['TRUE', '1.0']


def pick_device(base_device: str, gpu_id: int) -> str:
    if base_device.startswith("cuda"):
        return f"cuda:{gpu_id}"
    return base_device


def build_run_cmd(task_xlsx: str, task_steps: List[int], args, device: str) -> List[str]:
    cmd = [sys.executable, RUN_PATH, task_xlsx]
    cmd += ["--config", args.config]
    if args.data_root:
        cmd += ["--data_root", args.data_root]
    cmd += ["--steps", ",".join(str(s) for s in task_steps)]
    cmd += ["--device", device]
    cmd += ["--mode", args.mode]

    if args.force_all:
        cmd.append("--force_all")
    if args.log_file:
        cmd += ["--log_file", args.log_file]
    if args.check:
        cmd.append("--check")
    if args.clean:
        cmd += ["--clean", args.clean]
    if args.clean_dry_run:
        cmd.append("--clean_dry_run")

    return cmd


def run_single_process(args, gpu_ids: List[int]) -> int:
    device = pick_device(args.device, gpu_ids[0])
    steps = parse_steps(args.steps)
    cmd = build_run_cmd(args.xlsx, steps, args, device=device)
    print(f"[mp] single-process mode, forwarding to run_stages.py with device={device}")
    print("[mp] cmd:", " ".join(cmd))
    return subprocess.call(cmd)


def lock_paths_for_task(task: Task) -> List[str]:
    if task.kind == "scene":
        base = os.path.join(task.scene_folder_abs, ".locks")
    else:
        base = os.path.join(task.scene_folder_abs, task.seq_name, ".locks")
    return [os.path.join(base, f"step_{step}.lock") for step in sorted(task.steps)]


def try_acquire_locks(lock_paths: List[str]):
    if fcntl is None:
        return None
    fds = []
    try:
        for lp in lock_paths:
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                for got in fds:
                    fcntl.flock(got, fcntl.LOCK_UN)
                    os.close(got)
                return None
            fds.append(fd)
        return fds
    except Exception:
        for got in fds:
            try:
                fcntl.flock(got, fcntl.LOCK_UN)
                os.close(got)
            except Exception:
                pass
        raise


def release_locks(fds):
    if not fds or fcntl is None:
        return
    for fd in fds:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def worker_loop(worker_id: int, gpu_id: int, task_q: mp.Queue, result_q: mp.Queue, args_dict: Dict):
    class Obj:
        pass

    args = Obj()
    for k, v in args_dict.items():
        setattr(args, k, v)

    while True:
        try:
            task_dict = task_q.get(timeout=args.worker_poll_interval)
        except queue.Empty:
            continue

        if task_dict is None:
            result_q.put({"type": "worker_exit", "worker_id": worker_id})
            return

        task = Task(**task_dict)
        lock_fds = None
        try:
            lock_paths = lock_paths_for_task(task)
            lock_fds = try_acquire_locks(lock_paths)
            if lock_fds is None:
                result_q.put({
                    "type": "task_done",
                    "task_id": task.task_id,
                    "status": "skipped_locked",
                    "retry": task.retry,
                    "worker_id": worker_id,
                })
                continue

            device = pick_device(args.device, gpu_id)
            cmd = build_run_cmd(task.xlsx_path, task.steps, args, device=device)
            print(f"[mp][worker-{worker_id}][gpu-{gpu_id}] running task={task.task_id} kind={task.kind} steps={task.steps}")
            ret = subprocess.call(cmd)
            status = "success" if ret == 0 else "failed"
            result_q.put({
                "type": "task_done",
                "task_id": task.task_id,
                "status": status,
                "retry": task.retry,
                "ret": ret,
                "worker_id": worker_id,
                "task": task_dict,
            })
        except Exception as e:
            result_q.put({
                "type": "task_done",
                "task_id": task.task_id,
                "status": "failed",
                "retry": task.retry,
                "ret": -1,
                "error": str(e),
                "worker_id": worker_id,
                "task": task_dict,
            })
        finally:
            release_locks(lock_fds)


def write_task_xlsx(df_rows: pd.DataFrame, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, f"{name}.xlsx")
    df_rows.to_excel(path, index=False)
    return path


def build_tasks(df: pd.DataFrame, args, steps: List[int], task_dir: str):
    scene_steps = [s for s in steps if s in (1, 2, 3)]
    seq_steps = [s for s in steps if s >= 4]

    rows = []
    for _, row in df.iterrows():
        scene_folder_rel = str(row['scene_folder'])
        scene_folder_abs = os.path.join(args.data_root, scene_folder_rel) if args.data_root else scene_folder_rel
        seq_name = str(row['seq_name'])
        failed = get_bool_from_excel(row, "FAILED")
        rows.append((scene_folder_rel, scene_folder_abs, seq_name, failed, row))

    scene_map: Dict[Tuple[str, str], List[pd.Series]] = {}
    for scene_rel, scene_abs, _seq, _failed, row in rows:
        key = (scene_rel, scene_abs)
        scene_map.setdefault(key, []).append(row)

    scene_tasks: List[Task] = []
    if scene_steps:
        for idx, ((scene_rel, scene_abs), row_list) in enumerate(sorted(scene_map.items(), key=lambda x: x[0][0])):
            df_scene = pd.DataFrame(row_list)
            xlsx_path = write_task_xlsx(df_scene, task_dir, f"scene_{idx}")
            scene_tasks.append(Task(
                task_id=f"scene::{scene_rel}",
                kind="scene",
                scene_folder_rel=scene_rel,
                scene_folder_abs=scene_abs,
                seq_name="",
                steps=scene_steps,
                xlsx_path=xlsx_path,
                retry=0,
            ))

    seq_tasks: List[Task] = []
    if seq_steps:
        seq_idx = 0
        for scene_rel, scene_abs, seq_name, failed, row in rows:
            if failed and not args.force_all:
                continue
            df_seq = pd.DataFrame([row])
            xlsx_path = write_task_xlsx(df_seq, task_dir, f"seq_{seq_idx}")
            seq_idx += 1
            seq_tasks.append(Task(
                task_id=f"seq::{scene_rel}/{seq_name}",
                kind="seq",
                scene_folder_rel=scene_rel,
                scene_folder_abs=scene_abs,
                seq_name=seq_name,
                steps=seq_steps,
                xlsx_path=xlsx_path,
                retry=0,
            ))

    return scene_tasks, seq_tasks


def run_task_pool(tasks: List[Task], args, gpu_ids: List[int], stage_name: str) -> bool:
    if not tasks:
        print(f"[mp] {stage_name}: no tasks")
        return True

    task_q = mp.Queue()
    result_q = mp.Queue()

    args_dict = vars(args).copy()
    workers: List[mp.Process] = []
    for wid, gid in enumerate(gpu_ids):
        p = mp.Process(target=worker_loop, args=(wid, gid, task_q, result_q, args_dict), daemon=True)
        p.start()
        workers.append(p)

    for t in tasks:
        task_q.put(t.__dict__)

    total = len(tasks)
    done = 0
    success = 0
    failed = 0

    while done < total:
        msg = result_q.get()
        if msg.get("type") != "task_done":
            continue

        status = msg["status"]
        task_id = msg["task_id"]
        retry = msg.get("retry", 0)

        if status == "success":
            success += 1
            done += 1
            print(f"[mp] [{stage_name}] done {done}/{total}: {task_id}")
        elif status == "skipped_locked":
            done += 1
            print(f"[mp] [{stage_name}] locked-skip {done}/{total}: {task_id}")
        else:
            if retry < args.max_retries:
                task_dict = msg["task"]
                task_dict["retry"] = retry + 1
                task_q.put(task_dict)
                print(f"[mp] [{stage_name}] retry {retry + 1}/{args.max_retries}: {task_id}")
            else:
                failed += 1
                done += 1
                print(f"[mp] [{stage_name}] FAILED {done}/{total}: {task_id} (ret={msg.get('ret')})")

    for _ in workers:
        task_q.put(None)
    for p in workers:
        p.join(timeout=5)

    print(f"[mp] [{stage_name}] summary: total={total}, success={success}, failed={failed}, skipped_locked={total-success-failed}")
    return failed == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('xlsx', nargs='?', default=None, help='')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file')
    parser.add_argument('--steps', type=str, default='0')
    parser.add_argument('--data_root', type=str, default=None, help='')
    parser.add_argument('--clean', type=str, choices=['standard', 'fast', 'all'], help='forward to run_stages.py clean mode')
    parser.add_argument('--device', type=str, default='cuda', help='device base, e.g. cuda')
    parser.add_argument('--mode', type=str, default='skip', choices=['overwrite', 'skip'], help='overwrite or skip existing sequences')
    parser.add_argument('--log_file', type=str, default=None, help='log file')
    parser.add_argument('--check', action='store_true', help='check completion status of specified steps')
    parser.add_argument('--clean_dry_run', action='store_true', help='preview clean actions without deleting files')
    parser.add_argument('--force_all', action='store_true', help='force process/check sequences including FAILED rows')

    parser.add_argument('--gpu_ids', type=str, default=None, help='GPU ids, e.g. 0,1,2. multi-GPU auto-enabled when count>1')
    parser.add_argument('--worker_poll_interval', type=float, default=1.0, help='worker queue poll interval in seconds')
    parser.add_argument('--max_retries', type=int, default=1, help='max retries per task')

    # Help short-circuit: always show help and exit, regardless of other args.
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if not args.xlsx and not args.clean:
        print('Usage: python run_stages_mp.py <xlsx> [--steps 1-16] [--gpu_ids 0,1,2]')
        sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    steps = parse_steps(args.steps)

    # Keep original behavior for non-pipeline actions and single-GPU mode.
    if args.check or args.clean:
        code = run_single_process(args, gpu_ids)
        sys.exit(code)

    if len(gpu_ids) <= 1:
        code = run_single_process(args, gpu_ids)
        sys.exit(code)

    if fcntl is None:
        print('[mp] ERROR: fcntl not available, multi-process lock cannot be enabled.')
        sys.exit(2)

    print(f"[mp] multi-worker mode enabled: gpu_ids={gpu_ids}, lock=ON, max_retries={args.max_retries}")

    df = load_xlsx_rows(args.xlsx)

    task_dir = tempfile.mkdtemp(prefix='run_stages_mp_tasks_')
    print(f"[mp] task temp dir: {task_dir}")

    scene_tasks, seq_tasks = build_tasks(df, args, steps, task_dir)
    print(f"[mp] tasks built: scene={len(scene_tasks)}, seq={len(seq_tasks)}")

    ok_scene = run_task_pool(scene_tasks, args, gpu_ids, stage_name='scene')
    if not ok_scene:
        print('[mp] scene stage failed, aborting seq stage')
        sys.exit(3)

    ok_seq = run_task_pool(seq_tasks, args, gpu_ids, stage_name='seq')
    if not ok_seq:
        sys.exit(4)

    print('[mp] all tasks completed successfully')


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
