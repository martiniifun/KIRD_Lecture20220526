"""
SK텔레콤의 코바트KoBart 모델
유사한 국내 모델로는 네이버의 하이퍼클로바와
그 유명한 카카오브레인의 KoGPT3 등이 있다.
"""
from transformers import PreTrainedTokenizerFast
from tokenizers import SentencePieceBPETokenizer
from transformers import BartForConditionalGeneration
import streamlit as st
import torch



def tokenizer():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')
    return tokenizer


@st.cache(allow_output_mutation=True)
def get_model():
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization')
    model.eval()
    return model


default_text = '''
윤석열 대통령은 오는 18일 이원석 대검 차장검사를 새 정부 초대 검찰총장으로 지명할 것으로 알려졌다.
총장후보추천위는 지난 16일 이 차장과 여환섭 법무연수원장, 김후곤 서울고검장, 이두봉 대전고검장을 총장 후보로 선정했으며, 한동훈 법무부 장관은 이 차장을 윤 대통령에게 제청하기로 했다.
여권 핵심 관계자는 17일 연합뉴스와 통화에서 "특수통인 이 차장이 검찰총장으로 낙점된 것으로 안다"고 말했다.
다른 관계자는 통화에서 "총장 인선이 늦어진 만큼 검찰 조직의 신속한 안정을 위해 대검 차장을 총장으로 올리는 방안이 고려된 측면도 있다"고 설명했다.
이 차장은 전남 보성 출신으로 검찰 내 대표적인 특수통으로 분류된다.
대검 수사지원과장과 수사지휘과장, 서울중앙지검 특수1부장, 대검 기획조정부장, 제주지검장을 거쳐 현재 검찰총장 직무대리를 맡고 있다.
총장이 공석이 된 지난 5월부터 조직을 안정적으로 관리하고, 주요 사건 수사를 원활하게 지휘해왔다는 평가를 받는다. 한 장관과 검찰 인사도 긴밀히 상의해왔다.
다만, 사법연수원 27기로 경쟁자들보다 기수가 낮다는 점에서 파격 인사라는 분석도 나온다.
이 차장은 윤 대통령의 지명 이후 국회 법제사법위원회 인사청문회를 거쳐 최종 임명되게 된다. 국회 임명 동의는 필요로 하지 않는다.
'''


model = get_model()
tokenizer = tokenizer()
st.title("Summarization Model Test")
text = st.text_area("Input news :", value=default_text)

st.markdown("## Original News Data")
st.write(text)

if text:
    st.markdown("## Predict Summary")
    with st.spinner('processing..'):
        raw_input_ids = tokenizer.encode(text)
        input_ids = [tokenizer.bos_token_id] + \
            raw_input_ids + [tokenizer.eos_token_id]
        summary_ids = model.generate(torch.tensor([input_ids]),
                                     max_length=512,
                                     early_stopping=True,
                                     repetition_penalty=2.0)
        summ = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    st.write(summ)