"""
Shared utilities for reading and processing xlsx manifest files.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def get_bool_from_excel(row, key: str, default: bool = False) -> bool:
    """
    Extract boolean value from Excel row.
    Handles various representations: True/False, 1/0, 'true'/'false', etc.
    """
    if key not in row:
        return default
    val = row[key]
    if pd.isna(val):
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.lower() in ('true', '1', 'yes', 'y')
    return default


def read_xlsx_all_sheets(xlsx_path: str) -> pd.DataFrame:
    """
    Read xlsx file, concatenating all sheets if multiple exist.
    
    Args:
        xlsx_path: Path to xlsx file
        
    Returns:
        DataFrame with all rows from all sheets
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")
    
    xl = pd.ExcelFile(xlsx_path)
    if len(xl.sheet_names) > 1:
        print(f"Found {len(xl.sheet_names)} sheets in {xlsx_path}: {xl.sheet_names}")
        dfs = []
        for sheet_name in xl.sheet_names:
            df_sheet = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            print(f"  Sheet '{sheet_name}': {len(df_sheet)} rows")
            dfs.append(df_sheet)
        df = pd.concat(dfs, ignore_index=True)
        print(f"Total rows: {len(df)}")
    else:
        df = pd.read_excel(xlsx_path)
        print(f"Single sheet: {len(df)} rows")
    
    return df


def filter_failed_rows(df: pd.DataFrame, skip_failed: bool = True) -> pd.DataFrame:
    """
    Filter out rows marked as FAILED.
    
    Args:
        df: Input DataFrame
        skip_failed: If True, filter out FAILED rows
        
    Returns:
        Filtered DataFrame
    """
    if not skip_failed:
        return df
    
    if 'FAILED' in df.columns:
        original_len = len(df)
        df = df[df['FAILED'].isna() | (df['FAILED'] == '')]
        filtered = original_len - len(df)
        if filtered > 0:
            print(f"Filtered out {filtered} FAILED rows")
    
    return df


def build_scene_seq_list(
    xlsx_path: str,
    data_root: Optional[str] = None,
    skip_failed: bool = True,
    require_optim_params: bool = False
) -> List[Tuple[str, str, Path]]:
    """
    Build a list of (scene_folder_rel, seq_name, seq_path) from xlsx.
    
    Args:
        xlsx_path: Path to xlsx file
        data_root: Optional root directory to prefix scene_folder
        skip_failed: Skip rows marked as FAILED
        require_optim_params: Only include sequences with optim_params.npz
        
    Returns:
        List of tuples: (scene_folder_rel, seq_name, seq_path_absolute)
    """
    df = read_xlsx_all_sheets(xlsx_path)
    df = filter_failed_rows(df, skip_failed)
    
    result = []
    missing_seq = 0
    missing_optim = 0
    
    for idx, row in df.iterrows():
        scene_folder_rel = str(row.get('scene_folder', '')).strip()
        seq_name = str(row.get('seq_name', '')).strip()
        
        if not scene_folder_rel or not seq_name:
            continue
        
        if data_root:
            scene_folder_abs = Path(data_root) / scene_folder_rel
        else:
            scene_folder_abs = Path(scene_folder_rel)
        
        seq_path = scene_folder_abs / seq_name
        
        if not seq_path.exists():
            missing_seq += 1
            continue
        
        if require_optim_params:
            optim_path = seq_path / "optim_params.npz"
            if not optim_path.exists():
                missing_optim += 1
                continue
        
        result.append((scene_folder_rel, seq_name, seq_path))
    
    if missing_seq > 0:
        print(f"Warning: {missing_seq} sequences not found on disk")
    if missing_optim > 0:
        print(f"Warning: {missing_optim} sequences missing optim_params.npz")
    
    print(f"Found {len(result)} valid sequences")
    return result


def build_manifest_for_viser(
    xlsx_path: str,
    data_root: Optional[str] = None
) -> Dict:
    """
    Build manifest structure for visualize_viser.py.
    
    Returns:
        Dict mapping scene_key to scene info with seq_items list
    """
    seq_list = build_scene_seq_list(
        xlsx_path, 
        data_root, 
        skip_failed=True, 
        require_optim_params=True
    )
    
    manifest = {}
    
    for scene_rel, seq_name, seq_path in seq_list:
        scene_path = seq_path.parent
        scene_key = scene_rel
        
        if scene_key not in manifest:
            manifest[scene_key] = {
                "scene_path": scene_path,
                "scene_label": scene_path.name,
                "seq_items": []
            }
        
        manifest[scene_key]["seq_items"].append((seq_name, seq_path))
    
    return manifest
