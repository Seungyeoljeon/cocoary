__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#import
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever


import base64
import json
import http.client


class CompletionExecutor:
    def __init__(self, host, api_key, api_key_primary_val, request_id):
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id

    def _send_request(self, completion_request):
        headers = {
            'Content-Type': 'application/json; charset=utf-8',
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id
        }

        conn = http.client.HTTPSConnection(self._host)
        conn.request('POST', '/testapp/v1/completions/LK-D2', json.dumps(completion_request), headers)
        response = conn.getresponse()
        result = json.loads(response.read().decode(encoding='utf-8'))
        conn.close()
        return result

    def execute(self, completion_request):
        res = self._send_request(completion_request)
        if res['status']['code'] == '20000':
            return res['result']['text']
        else:
            return 'Error'



completion_executor = CompletionExecutor(
    host='clovastudio.apigw.ntruss.com',
    api_key='NTA0MjU2MWZlZTcxNDJiY9Zxd0Yp8yO0KRGL/G2THa83nyLxrWOgQGh5j8+XuTW0hfGBTfPfii1P2N7TKAodSX0GI8BVPSNQWtIiKS9Nnhfjr3hlF6aQegJB/KktvQ8a2YfUsx3DvEYe6qclZCvk9zqnRnKWrAJ4d/iN1RtzPJsYAZ5HuUiWYKtB1iMtLy/e5rEuraNaRMke5fvv2+VHuM1R2JNKNrcIbfWVfvsZbs4=',
    api_key_primary_val = 'psYV8l6gXowugYMsw4BEzW01IqtOimhrNsK0itQW',
    request_id='74915296c74046429150d87d005cee06'
)





#Stream 받아 줄 Hander 만들기
class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)



# from langchain.llms import CTransformers
chat_model = ChatOpenAI(model="gpt-4", temperature=0)
# llm = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q2_K.bin",
#     model_type="llama"
# )
button(username="sfix", floating=True, width=221)


st.title('코코어리!')
st.caption('나를 가장 잘 알아주는 내 친구 코코와 일상을 기록해보세요!')

import openai
import streamlit as st

# def extract_emotion_from_response_korean(response):
#     Some basic keywords for emotions (this can be expanded or refined)
#     emotions = ["happy", "sad", "angry", "surprised", "joyful", "confused", "excited", "worried"]
    
#     Check which emotion keyword is present in the response
#     for emotion in emotions:
#         if emotion in response.lower():
#             return emotion
#     return "neutral"



# def generate_dalle_prompt_with_gpt_emotion(response, gpt_model="gpt-3.5-turbo"):
#     # Asking GPT about the main emotion of the response
#     emotion_query = {
#         "model": gpt_model,
#         "messages": [{"role": "system", "content": "You are a helpful emotion extraction assistant."},
#                     {"role": "user", "content": response},
#                     {"role": "assistant", "content": "Based on the above content, what is the main emotion expressed?"}]
#     }
#     emotion_response = openai.ChatCompletion.create(**emotion_query)
#     emotion = emotion_response['choices'][0]['message']['content']
    
#     return f"{emotion.capitalize()} Welsh Corgi named Coco"

def generate_dalle_prompt_from_emotion(emotion):
    return f"Coco, a Welsh corgi dog who communicates with her owner and expresses feelings like {emotion.capitalize()}"

def generate_dalle_image(prompt):
    """Generate an image using DALL-E-3 based on the given prompt."""
    # Creating image using DALL-E-3 with the given prompt
    response = openai.Image.create(prompt=prompt, n=1, size="256x256")
    return response['data'][0]['url']


col1, col2 = st.columns(2)
with col1:
    life = st.text_area('당신의 오늘 하루는 무슨일이 있었나요?', help='오늘 있었던 일을 말해주세요')

with col2:
    description = st.text_area('오늘 기분은 어땠어요?', help='어떤 기분인지 설명해주세요')


st.title("💬 코코와 대화하기")
st.caption("오늘 무슨 일 있었어?")
start_interview = st.button('코코와 하루 돌아보기')


# 첫번째 메시지 생성
if start_interview:
    if not life or not description:  # life, description이 비어있는지 확인
        st.warning("오늘 있었던 일과 기분을 입력해주세요.")
    else:
        interviewer = life + description + "위 내용을 참고해서 일기를 쓰도록 돕는 웰시코기 강아지 친구 역할을 수행한다. 답변은 한글로 한다. 오늘 있었던 일과와 기분에 대해서 강아지처럼 물어본다. 너의 이름은 '코코'이다. 반갑게 인사부터 해줘. 나는 프로그램이라서 라고 거절하지 않도록 한다."
        st.session_state["messages"] = [{"role": "user", "content": interviewer}]
        st.session_state["started"] = True  # 면접 시작 상태를 저장합니다.

        # 챗봇의 응답을 생성합니다.
    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        emotion = completion_executor.execute(msg['content'])
        prompt = generate_dalle_prompt_from_emotion(emotion)
        image_url = generate_dalle_image(prompt)
        st.image(image_url, caption='코코의 오늘 감정', use_column_width="auto")
        st.write(prompt)
    except Exception as e:
        st.write("에러", str(e))

# 사용자로부터 입력을 받습니다.
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # 챗봇의 응답을 생성합니다.
    try:
        response = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state.messages)
        msg = response.choices[0].message
        st.session_state.messages.append(msg)
        emotion = completion_executor.execute(msg['content'])
        prompt = generate_dalle_prompt_from_emotion(emotion)
        image_url = generate_dalle_image(prompt)
        st.image(image_url, caption='코코의 오늘 감정', use_column_width="auto")
        st.write(prompt)
    except Exception as e:
        st.write("에러", str(e))



if "started" in st.session_state and st.session_state["started"]:
    for message in st.session_state.get("messages", [])[1:]:
        st.chat_message(message["role"]).write(message["content"])

st.write(emotion)
st.write(prompt)