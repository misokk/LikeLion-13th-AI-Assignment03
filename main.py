import os
import json
from dotenv import load_dotenv, find_dotenv 
#.env 파일에서 환경변수 불러오기 위한 모듈
from openai import OpenAI
#Tpgether API와 호환되는 openai 패키지
import tiktoken
#텍스트를 토큰으로 변환하기 위한 라이브러리

_ = load_dotenv(find_dotenv())
#.env 파일을 찾아서 환경변수를 로드

API_KEY = os.environ["API_KEY"]
SYSTEM_MESSAGE = os.environ["SYSTEM_MESSAGE"]
#API키와 SYSTEM_MESSAGE 가져오기 
BASE_URL = "https://api.together.xyz"
#Together API 기본 URL 설정 
DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
#사용할 모델 이름 
FILENAME = "message_history.json"
#대화 기록 JSON 파일 이름 
INPUT_TOKEN_LIMIT = 2048
#입력 토큰 제한 설정 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
#Together API와 통신하기 위한 클라이언트 인스턴스 생성

#메시지를 전송하고 응답을 반환하는 함수
def chat_completion(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model, #사용 모델 지정
        messages=messages, #메시지 목록 전달
        temperature=temperature, #창의성(0.1~1.0)
        stream=False, #스트리밍 사용 여부 (False면 전체 응답 반환)
        **kwargs, #기타 옵션 전달
    )
    return response.choices[0].message.content #응답 중 첫 번째 메시지 내용 반환

#메시지를 전송하고 응답을 스트리밍 형태로 출력하는 함수
def chat_completion_stream(messages, model=DEFAULT_MODEL, temperature=0.1, **kwargs):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True, #스트리밍 모드 활성화
        **kwargs,
    )
    #스트리밍모드- 대형 언어 모델 API에서 응답을 조금씩 나눠서(real-time으로) 보내주는 방식, 조각조각 응답을 받을 수 있어 더 빠르게 출력

    response_content = ""  #전체 응답을 저장할 문자열

    for chunk in response:
        chunk_content = chunk.choices[0].delta.content #스트리밍 응답 조각
        if chunk_content is not None:
            print(chunk_content, end="") #콘솔에 실시간 출력
            response_content += chunk_content #전체 문자열에 추가

    print() #줄 바꿈
    return response_content #전체 응답 반환


#주어진 텍스트가 몇 개의 토큰으로 구성되었는지 계산
def count_tokens(text, model):
    encoding = tiktoken.get_encoding("cl100k_base") #토큰 인코딩 방식 설정
    tokens = encoding.encode(text) #텍스트를 토큰으로 인코딩
    return len(tokens) #토큰 수 반환

#메시지 전체의 토큰 수를 계산
def count_total_tokens(messages, model):
    total = 0
    for message in messages:
        total += count_tokens(message["content"], model)
    return total

#토큰 수가 제한을 초과하지 않도록 오래된 메시지를 제거
def enforce_token_limit(messages, token_limit, model=DEFAULT_MODEL):
    while count_total_tokens(messages, model) > token_limit:
        if len(messages) > 1:
            messages.pop(1) #가장 오래된 user/assistant 메시지를 제거 (index 1)
        else:
            break

#객체를 JSON 파일로 저장
def save_to_json_file(obj, filename):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"{filename} 파일에 내용을 저장하는 중에 오류가 발생했습니다:\n{e}")

#JSON 파일에서 데이터를 읽어오기
def load_from_json_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        print(f"{filename} 파일 내용을 읽어오는 중에 오류가 발생했습니다:\n{e}")
        return None

#챗봇을 실행하는 메인 함수
def chatbot():
    #이전 대화 기록 불러오기
    messages = load_from_json_file(FILENAME)
    if not messages:
        #기록이 없으면 system 메시지부터 시작
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
        ]

    print("Chatbot: 안녕하세요! 무엇을 도와드릴까요? (종료하려면 'quit' 또는 'exit'을 입력하세요.)\n")
    
    while True:
        user_input = input("You: ") #사용자 입력 받기
        if user_input.lower() in ['quit', 'exit']: #종료 조건
            break

        messages.append({"role": "user", "content": user_input}) #입력 메시지 추가

        total_tokens = count_total_tokens(messages, DEFAULT_MODEL) #현재 총 토큰 수 계산
        print(f"[현재 토큰 수: {total_tokens} / {INPUT_TOKEN_LIMIT}]")

        enforce_token_limit(messages, INPUT_TOKEN_LIMIT) #토큰 제한 초과 시 메시지 제거

        print("\nChatbot: ", end="")
        response = chat_completion_stream(messages) #응답 스트리밍 출력
        print()

        messages.append({"role": "assistant", "content": response}) #응답 메시지 저장

        save_to_json_file(messages, FILENAME) #대화 내용 파일에 저장

#챗봇 함수 실행
chatbot()