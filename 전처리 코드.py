import pandas as pd
import spacy
import re
import time
import warnings
from symspellpy import SymSpell, Verbosity
import wordninja
from keybert import KeyBERT

gpu_available = spacy.prefer_gpu()
if gpu_available:
    print("GPU를 사용하고 있습니다.")
else:
    print("GPU를 사용하지 않고 있습니다.")

warnings.simplefilter(action='ignore', category=FutureWarning)


# 숫자 및 기호 제거 함수
def remove_numbers_and_signs(text):
    if not isinstance(text, str):
        return ""
    cleaned_string = ''.join([char for char in text if char.isalpha() or char.isspace()])
    text = re.sub(r'\S*\d\S*', '', text)
    return re.sub(r'[^A-Za-z\s]', '', cleaned_string)


# 불용어 및 색깔 제거 함수
def remove_stopwords_and_colors(text, stopwords, colors, nlp):
    if not isinstance(text, str) or text is None:
        return ""
    text=text.lower()
    doc = nlp(text)
    return " ".join(
        [token.text for token in doc if token.text not in stopwords and token.text not in colors and token.is_alpha]
    )


# SymSpell과 WordNinja를 사용한 철자 및 띄어쓰기 수정 함수
def correct_spelling_and_spacing(text, sym_spell):
    if not isinstance(text, str) or text is None:
        return ""

    # Step 1: Wordninja로 띄어쓰기 수행
    words = wordninja.split(text)
    corrected_words = []

    # Step 2: SymSpell로 오타 교정 수행
    for word in words:
        if word.isalpha():
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=1)
            corrected_word = suggestions[0].term if suggestions else word
        else:
            corrected_word = word
        corrected_words.append(corrected_word)

    return " ".join(corrected_words)

# 형용사 및 명사만 추출하는 함수
def extract_individual_words(text, nlp):
    if pd.isna(text) or text.strip() == "":
        return text

    if not isinstance(text, str):
        return ""

    doc = nlp(text)

    individual_words = set(
        token.text for token in doc
        if token.pos_ in ['NOUN', 'ADJ','VERB'] and token.is_alpha
    )

    return " ".join(sorted(individual_words))


# KeyBERT를 사용한 키워드 추출 함수
def extract_keywords_with_keybert(text, kw_model, top_n= 7 ):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 1)
                                         ,use_mmr=True, top_n=top_n, diversity=0.3)
    return " ".join([kw[0] for kw in keywords])


# 데이터 전처리 함수
def preprocess_descriptions_with_keybert(df, column, stopwords, colors, nlp, sym_spell, kw_model):
    start_time = time.time()

    # 1. 숫자 및 기호 제거
    df[column] = df[column].apply(remove_numbers_and_signs)
    print(f"Execution Time (Remove Numbers and Signs): {time.time() - start_time:.5f} seconds")

    # 2. 불용어 및 색깔 제거
    start_time = time.time()
    df[column] = df[column].apply(lambda x: remove_stopwords_and_colors(x, stopwords, colors, nlp))
    print(f"Execution Time (Remove Stopwords and Colors): {time.time() - start_time:.5f} seconds")

    # 3. 철자 및 띄어쓰기 수정
    start_time = time.time()
    df[column] = df[column].apply(lambda x: correct_spelling_and_spacing(x, sym_spell))
    print(f"Execution Time (Spelling + Spacing Correction): {time.time() - start_time:.5f} seconds")

    # 4. 불용어 및 색깔 제거 (다시 실행)
    start_time = time.time()
    df[column] = df[column].apply(lambda x: remove_stopwords_and_colors(x, stopwords, colors, nlp))
    print(f"Execution Time (Second Removal of Stopwords and Colors): {time.time() - start_time:.5f} seconds")

    # 5. 명사 및 형용사 추출 (spaCy 사용)
    start_time = time.time()
    df[column] = df[column].apply(lambda x: extract_individual_words(x, nlp))
    print(f"Execution Time (Noun-Adjective Extraction): {time.time() - start_time:.5f} seconds")

    # 6. KeyBERT를 사용한 키워드 추출
    start_time = time.time()
    df[column] = df[column].apply(lambda x: extract_keywords_with_keybert(x, kw_model))
    print(f"Execution Time (Keyword Extraction with KeyBERT): {time.time() - start_time:.5f} seconds")

    return df


# 메인 실행 부분
start_time = time.time()

# SymSpell 초기화
sym_spell = SymSpell(max_dictionary_edit_distance=1, prefix_length=9)
sym_spell.load_dictionary("C:/Users/APC/Desktop/frequency_dictionary_en_82_765.txt"
                          , term_index=0, count_index=1)

# spaCy 및 KeyBERT 초기화

nlp = spacy.load('en_core_web_lg', disable=['ner'])
kw_model = KeyBERT()

# 불용어 및 색깔 세트 정의
stopwords = set(
    ['a', 'an', 'the', 'hs', 'hscode', 'code', 'and', 'to', 'consignee', 'shipmentthc', 'of', 'kgs', 'tel', 'for', 'on',
     'in', 'rolls', 'mm', 'm', 'mil', 'gsn', 'inch', 'nos', 'ltr', 'thn', 'ton', 'tkw', 'kg', 'sqm', 'mtr', 'prs',
     'cum', 'crt', 'no', 'shipment', 'po', 'fee', 'fca', 'qty', 'c', 'x', 'lh', 'price', 'use', 'this', 'that',
     'invoice', 'kgm', 'fax', 'box', 'email', 'inthe', 'is', 'as', 'hts', 'total', 'per', 'instructions', 'etc', 'e',
     'd', 'hscodes', 'b', 'ss', 'f', 'r', 'l', 'ff', 'tt', 'shipper', 'south', 's', 'shippers', 'w', 'n', 'xl', 'usa',
     'id', 'invo', 't', 'co', 'into', 'y', 'with', 'p', 'un', 'other', 'tax', 'vat', 'without', 'indonesia', 'or',
     'japan', 'there', 'by', 'be', 'o', 'china', 'xshanghai', 'quantity', 'hsco', 'sub', 'date', 'lspos', 'gbx', 'xl',
     'cfscfs', 'h', 'used', 'not', 'does', 'contains', 'shenzhen', 'ikea', 'toyota', 'year', 'nissan', 'viscoatex',
     'dal', 'tanzania', 'customer', 'shanghai', 'model', 'via', 'maximum', 'bywater', 'speed','pcs','dpi','bags','bag',
     'packing','freight','packages','prepaid','lts','ft','weight','behalf','net'
     ,'unit','containers','exf','lax','materials','items','cartons','retail','fp','thc','order','package','container','pallets','substance',' packaging','transferable'
     ,'material','bulk','pci','ref','same','docs','fc','com','web','declaration','packaging','collect','item','number','num','ct','ns','rb','st','tr','ml','reference',
     'pbs','la','import','importer','pc','assortment','ship','shipping','exp','information','exporter','pm','charges','act','art','address','contain','said','pkg',
     'harmonized','gur' ,'consisting', 'packed','units','including','freighted','sex'
     ,'provided'])
colors = set(['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'purple', 'gray', 'brown', 'gold', 'silver'])

# 데이터 로드
df = pd.read_csv('C:/Users/APC/Desktop/data/extracted_hs_codes_v2.csv', dtype='unicode', sep=',', encoding='utf-8-sig', on_bad_lines='skip', engine='python')
df['ORIGINAL_DESCRIPTION'] = df['PRODUCT_DESCRIPTION']


# 데이터 전처리
df = preprocess_descriptions_with_keybert(df, 'PRODUCT_DESCRIPTION', stopwords, colors, nlp, sym_spell, kw_model)

# 결과 저장
df.to_csv('C:/Users/APC/Desktop/US_PRO/us_processing_all.csv', encoding='UTF-8-sig', index=False)

execution_time = time.time() - start_time
print(f"Total Execution Time: {execution_time:.5f} seconds")
