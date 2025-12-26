import os
from pathlib import Path
from typing import Set, Tuple


def extract_file_keys(folder_path: str) -> Set[Tuple[str, str, str]]:
    """모든 파일의 (prefix, second, fourth) 추출"""
    file_keys = set()       # 중복 제거
    folder = Path(folder_path)
    
    for file_path in folder.iterdir():
        if file_path.is_file():
            split_list = file_path.stem.split("_")
            if len(split_list) > 3:
                file_keys.add((split_list[0], split_list[1], split_list[3]))
    
    return file_keys


# 경로 
STR_PATH = "../239.건축 도면 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_OBJ"
SPA_PATH = "../239.건축 도면 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_SPA"
OBJ_PATH = "../239.건축 도면 데이터/01-1.정식개방데이터/Training/02.라벨링데이터/TL_STR"

# 파일 키 세트
str_keys = extract_file_keys(STR_PATH)
spa_keys = extract_file_keys(SPA_PATH)
obj_keys = extract_file_keys(OBJ_PATH)  

# 교집합
common_all = str_keys & spa_keys & obj_keys
common_str_spa = str_keys & spa_keys
common_spa_obj = spa_keys & obj_keys
common_str_obj = str_keys & obj_keys

print(f"전체 공통: {len(common_all)}개")
print(f"STR+SPA 공통: {len(common_str_spa)}개") 
print(f"SPA+OBJ 공통: {len(common_spa_obj)}개")
print(f"STR+OBJ 공통: {len(common_str_obj)}개")
