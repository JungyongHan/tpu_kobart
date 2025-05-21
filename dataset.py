import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch_xla.core.xla_model as xm
pd.set_option('mode.chained_assignment', None)

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len=512, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file)
        self.newline_token = '<LF>'  # 토크나이저에 추가된 특수 토큰
        self.ignore_index = ignore_index
        newline_token_id = self.tokenizer.convert_tokens_to_ids(self.newline_token)
        if newline_token_id == self.tokenizer.unk_token_id:
            # 이 경우, 토큰이 토크나이저에게 인식되지 않는다는 의미입니다.
            print(f"Debug: self.newline_token = '{self.newline_token}'")
            print(f"Debug: Converted ID = {newline_token_id}")
            print(f"Debug: UNK ID = {self.tokenizer.unk_token_id}")
            print(f"Debug: self.tokenizer.added_tokens_decoder = {self.tokenizer.added_tokens_decoder}")
            raise ValueError(
                f"'{self.newline_token}' 토큰이 토크나이저에 의해 UNK 토큰으로 처리됩니다. "
                f"토큰이 올바르게 추가되었는지, 그리고 모델의 임베딩 레이어가 업데이트되었는지 확인하세요."
            )

        # 데이터 전처리
        self.docs = self.preprocess_data(self.docs)
        self.len = self.docs.shape[0]

    # 데이터 전처리 함수 정의
    def preprocess_data(self, data):
        # NaN 값 제거
        original_len = len(data)
        data['article'] = data['article'].astype(str)
        data['script'] = data['script'].astype(str)

        data['article_len'] = data['article'].apply(len)
        data['script_len'] = data['script'].apply(len)

        data = data[data['article_len'] > data['script_len']]
        filtered_len = len(data)
        if xm.is_master_ordinal():
            print(f"데이터 필터링: {original_len}개 중 {filtered_len}개 남음 ({original_len - filtered_len}개 제외)")
        
        def convert_br_to_blank(text):
            return text.replace('<br>', '')

        # <br> 태그를 실제 개행 문자로 변환
        def convert_br_to_newline(text):
            return text.replace('<br>', '\n')
        
        # HTML 태그 제거
        def clean_html(text):
            return re.sub(r'<[^>]+>', '', text)
        
        # 이메일 주소 제거
        def remove_email(text):
            return re.sub(r'\S+@\S+\.\S+', '', text)
        
        # 웹사이트 주소 제거
        def remove_urls(text):
            return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
        # 괄호 내용 제거 (소괄호, 대괄호, 중괄호)
        def remove_brackets(text):
            text = re.sub(r'\([^)]*\)', '', text)  # 소괄호
            text = re.sub(r'\[[^\]]*\]', '', text)  # 대괄호
            text = re.sub(r'\{[^}]*\}', '', text)  # 중괄호
            return text
        
        # 특수문자 제거 (말줄임표, 중간점 등)
        def remove_special_chars(text):
            # 특수문자 제거 (알파벳, 숫자, 한글, 공백, 개행문자 외 모두 제거)
            text = re.sub(r'[^\w\s가-힣\n]', '', text)
            return text
        
        # 여러 공백을 하나로 치환 (개행문자는 유지)
        def clean_spaces(text):
            text = re.sub(r' +', ' ', text)  # 여러 공백을 하나로
            text = re.sub(r'\n+', '\n', text)  # 여러 개행을 하나로
            return text.strip()

        # 전처리 적용 - <br>을 실제 개행문자로 변환
        data['article'] = data['article'].apply(convert_br_to_blank)
        data['script'] = data['script'].apply(convert_br_to_newline)
        
        # 나머지 전처리 적용
        data['article'] = data['article'].apply(clean_html)
        data['article'] = data['article'].apply(remove_email)
        data['article'] = data['article'].apply(remove_urls)
        data['article'] = data['article'].apply(remove_brackets)
        data['article'] = data['article'].apply(remove_special_chars)
        data['article'] = data['article'].apply(clean_spaces)

        data['script'] = data['script'].apply(clean_html)
        data['script'] = data['script'].apply(clean_spaces)
        
        data = data.drop(['article_len', 'script_len'], axis=1)
        data = data[data['article'].str.strip().astype(bool) & data['script'].str.strip().astype(bool)]
        return data

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] *(self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        instance['script'] = instance['script'].replace('\n', self.newline_token)
        input_ids = self.tokenizer.encode(instance['article'])
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(instance['script'])
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)
               }

    # def __getitem__(self, idx):
    #     instance = self.docs.iloc[idx]
        
    #     # 인코딩 통합 처리
    #     article = instance['article'].replace('\n', self.newline_token)
    #     script = instance['script'].replace('\n', self.newline_token)

    #     # 인코더 입력 처리 (최대 길이-2: [CLS], [SEP] 공간 보존)
    #     encoder_inputs = self.tokenizer(
    #         article,
    #         # max_length=self.max_len-2, add_special_tokens 추가로인해 주석처리
    #         max_length=self.max_len,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='np',
    #         add_special_tokens=True
    #     )

    #     # 디코더 입력 처리 (레이블 생성을 위한 별도 인코딩)
    #     decoder_inputs = self.tokenizer(
    #         script,
    #         # max_length=self.max_len-1,  # [EOS] 공간 보존 add_special_tokens 추가로 인해 주석처리
    #         max_length=self.max_len,
    #         padding='max_length',
    #         truncation=True,
    #         return_tensors='np',
    #         add_special_tokens=True
    #     )

    #     # 레이블 생성 (패딩 부분 -100으로 마스킹)
    #     labels = decoder_inputs['input_ids'][0].copy()
    #     labels[labels == self.tokenizer.pad_token_id] = self.ignore_index

    #     return {
    #         'input_ids': encoder_inputs['input_ids'][0].astype(np.int32),
    #         'attention_mask': encoder_inputs['attention_mask'][0].astype(np.float32),
    #         'decoder_input_ids': decoder_inputs['input_ids'][0].astype(np.int32),
    #         'labels': labels.astype(np.int32)
    #     }

    def __len__(self):
        return self.len