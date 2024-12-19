import pandas as pd
import re

# 파일 로드
file_path = 'C:/Users/APC/Desktop/data/Batang_us_import_product_description_2024.csv'
data = pd.read_csv(file_path, dtype='unicode', sep=',', encoding='cp949')

# HS 코드 추출 함수
def extract_hs_code_v2(description):
    # HS, HTS, HS CODE 등 다양한 표현을 포괄하는 패턴 정의 (숫자 6~10자리)
    match = re.search(r'(HS|HS CODE|HS TARIFF|HSCODE|HTS CO DE|HS-CODE|HARMONIZEDCODE|HTS|HT CODE|HS-NO|HS#)'
                      r'[:\s]*([0-9]{6,10})', description)
    return match.group(2) if match else None

# HS 코드만 추출
data['HS_CODE'] = data['PRODUCT_DESCRIPTION'].astype(str).apply(extract_hs_code_v2)

# HS 코드가 있는 데이터 저장
hs_code_data = data[['HS_CODE', 'PRODUCT_DESCRIPTION']].dropna()
output_path = 'C:/Users/APC/Desktop/extracted_hs_codes_v2.csv'
hs_code_data.to_csv(output_path, index=False, header=True)

print(f"HS 코드만 포함된 파일이 '{output_path}'에 저장되었습니다.")

# HS 코드가 없는 데이터 저장
no_hs_code_data = data[data['HS_CODE'].isna()]
output_path_no_hs = 'C:/Users/APC/Desktop/no_hs_codes.csv'
no_hs_code_data.to_csv(output_path_no_hs, index=False, header=True)

print(f"HS 코드가 없는 데이터가 '{output_path_no_hs}'에 저장되었습니다.")





