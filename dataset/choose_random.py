import random
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm


def make_sample(        
    n: int, 
    source_dir: str, 
    json_dir: str,
    target_source_dir: str, 
    target_json_dir: str
) -> dict:
    """
    원본 이미지 + JSON 파일 랜덤 샘플링
    반환: {'total': int, 'source': int, 'json': int}
    """
    # Path 객체 미리 생성
    paths = {
        'source': Path(source_dir), 
        'json': Path(json_dir),
        'target_source': Path(target_source_dir), 
        'target_json': Path(target_json_dir)
    }
    
    # 타겟 폴더 생성
    paths['target_source'].mkdir(parents=True, exist_ok=True)
    paths['target_json'].mkdir(parents=True, exist_ok=True)
    
    # 원본 이미지 목록 + 개수 조정
    files = [f.name for f in paths['source'].iterdir() if f.is_file()]
    if len(files) < n:
        print(f"Not enough files: Adjust to {len(files)}")
        n = len(files)
    
    # 원본 이미지 랜덤 추출
    selected_files = random.sample(files, n)        
    results = {'total': n, 'source': 0, 'json': 0}      # 초기화
    
    # 이미지 + JSON 복사
    for f_name in tqdm(selected_files, desc="Sampling process", unit="file"):
        # 1. 원본 이미지 복사
        shutil.copy2(paths['source'] / f_name, paths['target_source'] / f_name)
        results['source'] += 1
        
        # 2. 매칭 JSON 파일 복사
        json_file = find_json_file_fast(paths['json'], f_name)
        if json_file:
            shutil.copy2(paths['json'] / json_file, paths['target_json'] / json_file)
            results['json'] += 1
    
    print(f"Completed: {results['total']} samples")
    
    return results

def find_json_file_fast(folder: Path, ref_file_name: str) -> Optional[str]:
    """이미지명에 맞는 JSON 파일 찾기"""
    ref_stem = Path(ref_file_name).stem
    json_candidate = folder / f"{ref_stem}.json"
    
    return json_candidate.name if json_candidate.exists() else None




if __name__ == "__main__":
    n = 1
    
    source_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    json_dir = "../01-1.정식개방데이터/Validation/02.라벨링데이터/VL_STR"

    target_source_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    target_json_dir = "/../01-1.정식개방데이터/Validation/02.라벨링데이터/VL_STR"

    result = make_sample(
        n, source_dir, json_dir, target_source_dir, target_json_dir
    )
