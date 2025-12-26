import os
import random
import shutil
from tqdm import tqdm

def make_sample_with_all_folders(n, source_dir, mask_dir, clean_dir, json_dir, target_source_dir, target_mask_dir, target_clean_dir, target_json_dir):
    # 타겟 폴더가 없으면 생성
    for d in [target_source_dir, target_mask_dir, target_clean_dir, target_json_dir]:
        os.makedirs(d, exist_ok=True)

    # 원본 폴더 내 파일 목록 (파일명만)
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

    # 랜덤으로 n개 선택
    selected_files = random.sample(files, n)

    for f_name in tqdm(selected_files):
        # 원본 파일 복사
        shutil.copy(os.path.join(source_dir, f_name), os.path.join(target_source_dir, f_name))

        # mask 파일 복사 (확장자가 같지 않을 수 있으므로 mask 폴더 내 동일 이름 파일 찾기)
        mask_file = find_same_name_file(mask_dir, f_name)
        if mask_file:
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(target_mask_dir, mask_file))

        # clean 파일 복사
        clean_file = find_same_name_file(clean_dir, f_name)
        if clean_file:
            shutil.copy(os.path.join(clean_dir, clean_file), os.path.join(target_clean_dir, clean_file))

        # json 파일 복사
        json_file = find_same_name_file(json_dir, f_name, target_ext='.json')
        if json_file:
            shutil.copy(os.path.join(json_dir, json_file), os.path.join(target_json_dir, json_file))

    print(f'Selected file number: {len(selected_files)}')

def find_same_name_file(folder_path, ref_file_name, target_ext=None):
    """
    folder_path 내에서 ref_file_name과 같은 이름의 파일을 찾음.
    target_ext가 지정되면 해당 확장자로 변환하여 찾음.
    """
    ref_name_without_ext = os.path.splitext(ref_file_name)[0]

    # 타겟 확장자가 있을 경우 그 확장자로 파일명 구성
    if target_ext:
        candidate_name = ref_name_without_ext + target_ext
        if os.path.exists(os.path.join(folder_path, candidate_name)):
            return candidate_name
        else:
            return None
    else:
        # 확장자 다른 파일 가능성 있으므로 같은 이름을 가진 파일 탐색
        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[0] == ref_name_without_ext:
                return fname
        return None



if __name__ == "__main__":
    n = 1
    
    source_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    mask_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    clean_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    json_dir = "../01-1.정식개방데이터/Validation/02.라벨링데이터/VL_STR"

    target_source_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    target_mask_dir = "/../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    target_clean_dir = "../01-1.정식개방데이터/Validation/01.원천데이터/VS_STR"
    target_json_dir = "../01-1.정식개방데이터/Validation/02.라벨링데이터/VL_STR"

    make_sample_with_all_folders(
        n, source_dir, mask_dir, clean_dir, json_dir, target_source_dir, target_mask_dir, target_clean_dir, target_json_dir
        )



