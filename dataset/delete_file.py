import os


def delete_file(folder_path):
    cnt = 0 

    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # APT_FP_STR 파일만 보존
            if not file_name.startswith("APT_FP_STR"):
                if os.path.isfile(file_path):
                    cnt += 1
                    os.remove(file_path)
                    print(file_path)

    return print(f"Deleted file number : {cnt}")




# 실행
delete_file("../239.건축 도면 데이터/01-1.정식개방데이터/Training/01.원천데이터/TS_STR")