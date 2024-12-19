import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time
from torch.utils.data import DataLoader

torch.cuda.empty_cache()

# 모델 로드 (SBERT) - GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# GPU 사용 가능한지 확인
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using CPU.")

# 파일 로드
file_path = 'C:/Users/APC/Desktop/US_PRO/us_processing_all_no_hscode.csv'
df = (pd.read_csv(file_path, dtype='unicode', sep=',', encoding='UTF-8')
      .dropna(subset=['PRODUCT_DESCRIPTION']))
df_india = pd.read_csv('C:/Users/APC/Desktop/sample/india_processing.csv'
                       , dtype='unicode', sep=',', encoding='UTF-8').dropna(
    subset=['PRODUCT_DESCRIPTION'])

# 배치 크기 설정
batch_size = 128  # 데이터 크기에 따라 적절히 조정

# 시간 측정 시작
start_time = time.time()

# 임베딩 계산을 배치로 처리
def compute_embeddings(data, batch_size):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in dataloader:
        batch_embeddings = model.encode(batch, convert_to_tensor=True, device=device)
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

print("Embedding calculation started...")
embeddings_us = compute_embeddings(df['PRODUCT_DESCRIPTION'].tolist(), batch_size)
embeddings_india = compute_embeddings(df_india['PRODUCT_DESCRIPTION'].tolist(), batch_size)
print("Embedding calculation completed.")

# 코사인 유사도 계산 및 배치로 처리
results = []
print("Similarity calculation started...")
for batch_start in range(0, len(embeddings_us), batch_size):
    batch_end = min(batch_start + batch_size, len(embeddings_us))
    us_batch = embeddings_us[batch_start:batch_end]

    # 코사인 유사도 계산
    similarities = util.cos_sim(us_batch, embeddings_india)
    top_similarities, top_indices = similarities.topk(1, dim=1)

    # 결과 저장
    for i in range(top_similarities.size(0)):
        global_idx = batch_start + i  # 전체 데이터셋의 인덱스 계산
        sim_scores = top_similarities[i].cpu().numpy().tolist()
        hs_codes = [df_india['HS CODE'].iloc[idx] for idx in top_indices[i].cpu().numpy()]

        results.append({
            '원본데이터': df['ORIGINAL_DESCRIPTION'].iloc[global_idx],
            'KEYWORDS': df['PRODUCT_DESCRIPTION'].iloc[global_idx],
            '실제 HS_CODE': df['HS_CODE'].iloc[global_idx],
            '매칭HSCODE': hs_codes[0],
            '유사도': sim_scores[0]
        })

print("Similarity calculation completed.")

# 결과를 데이터프레임으로 변환
test_df = pd.DataFrame(results)

# 파일 저장
output_path = 'C:/Users/APC/Desktop/US_PRO/no_hscode_result_all.csv'
test_df.to_csv(output_path, encoding='UTF-8-sig', header=True, index=False)

# 실행 시간 측정
execution_time = time.time() - start_time
print(f"Execution Time: {execution_time:.5f} seconds")