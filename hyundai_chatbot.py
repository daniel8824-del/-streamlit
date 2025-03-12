import os  # 파일 경로 처리
import pickle  # 파일 저장 및 로드
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.chat_models import ChatOpenAI  # 챗봇 모델
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI 임베딩 모델
from langchain.vectorstores import SKLearnVectorStore  # scikit-learn 벡터 저장소
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

def load_vectorstore():
    """
    저장된 벡터 저장소를 로드하는 함수
    
    Returns:
        SKLearnVectorStore: 로드된 벡터 저장소
    """
    # 벡터 저장소 파일 경로 확인
    vectorstore_path = "sklearn_index/vectorstore.pkl"
    
    if not os.path.exists(vectorstore_path):
        print(f"'{vectorstore_path}' 파일이 존재하지 않습니다. 먼저 create_vectorstore.py를 실행하여 벡터 저장소를 생성해주세요.")
        return None
    
    # 벡터 저장소 로드 (pickle 사용)
    with open(vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    
    print("벡터 저장소가 성공적으로 로드되었습니다.")
    
    return vectorstore

def create_chatbot():
    """
    챗봇을 생성하는 함수
    
    Returns:
        ConversationalRetrievalChain: 생성된 챗봇
    """
    # 벡터 저장소 로드
    vectorstore = load_vectorstore()
    
    if vectorstore is None:
        return None
    
    # 대화 메모리 초기화
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # ChatOpenAI 모델 초기화
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # 사용할 모델
        temperature=0.2  # 응답의 창의성 정도 (0에 가까울수록 결정적인 응답)
    )
    
    # 대화형 검색 체인 생성
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True  # 디버깅을 위한 상세 출력
    )
    
    return chatbot

def chat_with_bot(chatbot, query):
    """
    챗봇과 대화하는 함수
    
    Args:
        chatbot: 대화할 챗봇
        query (str): 사용자 질문
    
    Returns:
        str: 챗봇의 응답
    """
    if chatbot is None:
        return "챗봇이 초기화되지 않았습니다. 먼저 create_vectorstore.py를 실행하여 벡터 저장소를 생성해주세요."
    
    # 챗봇에 질문하고 응답 받기
    response = chatbot({"question": query})
    
    return response["answer"]

if __name__ == "__main__":
    # 챗봇 생성
    chatbot = create_chatbot()
    
    if chatbot:
        print("현대자동차 챗봇이 준비되었습니다. 질문을 입력하세요. 종료하려면 'exit' 또는 '종료'를 입력하세요.")
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n질문: ")
            
            # 종료 조건 확인
            if user_input.lower() in ["exit", "종료"]:
                print("챗봇을 종료합니다.")
                break
            
            # 챗봇에 질문하고 응답 출력
            response = chat_with_bot(chatbot, user_input)
            print(f"\n답변: {response}") 