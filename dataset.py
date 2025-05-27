import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch_xla.core.xla_model as xm
pd.set_option('mode.chained_assignment', None)

class KoBARTSummaryDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file)
        # 개행문자를 위한 특수 토큰 정의
        self.newline_token = '<LF>'
        self.docs = self.preprocess_data(self.docs)
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

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
        
    def analyze_padding_distribution(self, num_samples=1000):
        """패딩 분포를 분석하는 메서드"""
        input_ratios = []
        label_ratios = []
        script_lengths = []
        
        sample_indices = np.random.choice(len(self.docs), min(num_samples, len(self.docs)), replace=False)
        
        for idx in sample_indices:
            instance = self.docs.iloc[idx]
            
            # 원본 길이
            original_script_len = len(self.tokenizer.encode(instance['script']))
            script_lengths.append(original_script_len)
            
            # 실제 데이터 생성
            data = self.__getitem__(idx)
            
            # 패딩 비율 계산
            input_padding_ratio = np.sum(data['input_ids'] == self.pad_index) / len(data['input_ids'])
            label_padding_ratio = np.sum(data['labels'] == self.ignore_index) / len(data['labels'])
            
            input_ratios.append(input_padding_ratio)
            label_ratios.append(label_padding_ratio)
        
        print(f"=== Padding Analysis (n={len(sample_indices)}) ===")
        print(f"Input padding ratio - Mean: {np.mean(input_ratios):.2%}, Std: {np.std(input_ratios):.2%}")
        print(f"Label padding ratio - Mean: {np.mean(label_ratios):.2%}, Std: {np.std(label_ratios):.2%}")
        print(f"Script length - Mean: {np.mean(script_lengths):.1f}, 95th percentile: {np.percentile(script_lengths, 95):.1f}")
        print(f"Max length setting: {self.max_len}")
        
        return {
            'input_padding_ratios': input_ratios,
            'label_padding_ratios': label_ratios,
            'script_lengths': script_lengths
        }

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

    def __len__(self):
        return self.len