import os
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI 임베딩 모델
from langchain.vectorstores import FAISS  # FAISS 벡터 저장소
from langchain.chat_models import ChatOpenAI  # OpenAI 챗 모델
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

def load_vectorstore():
    """
    저장된 벡터 저장소를 로드하는 함수
    
    Returns:
        FAISS: 로드된 벡터 저장소
    """
    # 벡터 저장소가 존재하는지 확인
    if not os.path.exists("faiss_index"):
        print("벡터 저장소가 존재하지 않습니다. create_vectorstore.py를 먼저 실행해주세요.")
        return None
    
    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()
    
    # 저장된 벡터 저장소 로드 (allow_dangerous_deserialization=True 옵션 추가)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("벡터 저장소가 로드되었습니다.")
    
    return vectorstore

def create_chatbot():
    """
    현대자동차 설명서 챗봇을 생성하는 함수
    
    Returns:
        ConversationalRetrievalChain: 생성된 챗봇 체인
    """
    # 벡터 저장소 로드
    vectorstore = load_vectorstore()
    if vectorstore is None:
        return None
    
    # 대화 기록을 저장할 메모리 초기화
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # 메모리에 저장할 출력 키 지정
    )
    
    # OpenAI 챗 모델 초기화 (gpt-4o로 변경)
    llm = ChatOpenAI(
        model_name="gpt-4o",  # gpt-3.5-turbo에서 gpt-4o로 변경
        temperature=0.2  # 낮은 temperature로 정확한 답변 유도
    )
    
    # 대화형 검색 체인 생성
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}  # 상위 3개의 관련 문서 검색
        ),
        memory=memory,
        return_source_documents=True,  # 소스 문서 반환
        output_key="answer"  # 출력 키 지정
    )
    
    return chatbot

def chat_with_bot():
    """
    사용자와 챗봇 간의 대화를 처리하는 함수
    """
    # 챗봇 생성
    chatbot = create_chatbot()
    if chatbot is None:
        return
    
    print("현대자동차 설명서 챗봇이 준비되었습니다. 질문을 입력하세요. 종료하려면 'q' 또는 'quit'를 입력하세요.")
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n질문: ")
        
        # 종료 조건 확인
        if user_input.lower() in ['q', 'quit', '종료']:
            print("챗봇을 종료합니다.")
            break
        
        # 빈 입력 처리
        if not user_input.strip():
            print("질문을 입력해주세요.")
            continue
        
        try:
            # 챗봇에 질문 전달 및 응답 받기 (invoke 메서드 사용)
            response = chatbot.invoke({"question": user_input})
            
            # 응답 출력
            print("\n답변:", response["answer"])
            
            # 참고 페이지만 간결하게 출력
            pages = [doc.metadata.get('page', 'N/A') for doc in response["source_documents"]]
            print(f"\n참고 페이지: {', '.join(map(str, pages))}")
        
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    chat_with_bot() 