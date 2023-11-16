
## EDA 예시코드



df =pd.read_csv('C:/Users/LUNABIT/Desktop/open/open/international_trade.csv',dtype='unicode',sep=',')

# 데이터 shape 파악
df.shape
# 데이터 통계량 파악
df.describe()

# 결측치 개수 파악
df.isnull().sum().to_frame('nan_count')

# 서브플롯 생성
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 6))

# train_df의 item 열에서 고유한 값을 가져와서 반복
unique_items = train_df['item'].unique()
for i, item in enumerate(unique_items):
    row, col = i // 3, i % 3  # 행 및 열 위치 계산
    current_ax = ax[row, col]  # 현재 서브플롯 얻기

    # item 및 price(원/kg) 필터링
    filtered_data = train_df[(train_df['item'] == item) & (train_df['price(원/kg)'] != 0)]

    # seaborn을 사용하여 히스토그램 그리기
    sns.histplot(data=filtered_data, x='price(원/kg)', hue='corporation', ax=current_ax)

    current_ax.set_title(f'Item: {item}')

# 불필요한 서브플롯 숨기기
for i in range(len(unique_items), 6):
    row, col = i // 3, i % 3
    ax[row, col].axis('off')

# 서브플롯 간 간격 조정
plt.tight_layout()

# 그래프 표시
plt.show()