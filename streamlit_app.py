import streamlit as st
import os
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI 임베딩 모델
from langchain.vectorstores import FAISS  # FAISS 벡터 저장소
from langchain.chat_models import ChatOpenAI  # OpenAI 챗 모델
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="현대자동차 설명서 챗봇",
    page_icon="🚗",
    layout="centered"
)

# 제목 및 설명
st.title("🚗 현대자동차 설명서 챗봇")
st.markdown("""
이 챗봇은 현대자동차 설명서를 기반으로 질문에 답변합니다.
RAG(Retrieval-Augmented Generation) 기술을 활용하여 PDF 형식의 설명서에서 관련 정보를 검색하고, 
이를 기반으로 정확한 답변을 생성합니다.
""")

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

if 'ready' not in st.session_state:
    st.session_state.ready = False

def load_vectorstore():
    """
    저장된 벡터 저장소를 로드하는 함수
    
    Returns:
        FAISS: 로드된 벡터 저장소
    """
    # 벡터 저장소가 존재하는지 확인
    if not os.path.exists("faiss_index"):
        st.error("벡터 저장소가 존재하지 않습니다. create_vectorstore.py를 먼저 실행해주세요.")
        return None
    
    with st.spinner("벡터 저장소를 로드 중입니다..."):
        try:
            # OpenAI 임베딩 모델 초기화
            embeddings = OpenAIEmbeddings()
            
            # 저장된 벡터 저장소 로드
            vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            st.success("벡터 저장소가 로드되었습니다.")
            
            return vectorstore
        except Exception as e:
            st.error(f"벡터 저장소 로드 중 오류가 발생했습니다: {e}")
            return None

def create_chatbot(vectorstore):
    """
    현대자동차 설명서 챗봇을 생성하는 함수
    
    Returns:
        ConversationalRetrievalChain: 생성된 챗봇 체인
    """
    # 대화 기록을 저장할 메모리 초기화
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # OpenAI 챗 모델 초기화
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.2
    )
    
    # 대화형 검색 체인 생성
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    
    return chatbot

# 사이드바에 챗봇 초기화 버튼
with st.sidebar:
    st.header("챗봇 설정")
    
    if st.button("챗봇 초기화"):
        vectorstore = load_vectorstore()
        if vectorstore:
            with st.spinner("챗봇을 초기화 중입니다..."):
                st.session_state.chatbot = create_chatbot(vectorstore)
                st.session_state.ready = True
                st.session_state.messages = []
                st.success("챗봇이 준비되었습니다!")
    
    st.markdown("---")
    st.markdown("### 예시 질문")
    example_questions = [
        "아반떼 엔진 오일은 어떻게 교체하나요?",
        "타이어 공기압은 얼마로 유지해야 하나요?",
        "타이어가 펑크났어. 해결책을 알려줘",
        "창문에 서리가 자꾸 껴"
    ]
    
    for q in example_questions:
        if st.button(q):
            if st.session_state.ready:
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
            else:
                st.warning("먼저 챗봇을 초기화해주세요.")

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    if not st.session_state.ready:
        st.warning("먼저 챗봇을 초기화해주세요.")
    else:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    # 챗봇에 질문 전달 및 응답 받기
                    response = st.session_state.chatbot.invoke({"question": prompt})
                    
                    # 참고 페이지 추출
                    pages = [doc.metadata.get('page', 'N/A') for doc in response["source_documents"]]
                    
                    # 응답 및 참고 페이지 표시
                    full_response = f"{response['answer']}\n\n**참고 페이지**: {', '.join(map(str, pages))}"
                    message_placeholder.markdown(full_response)
                    
                    # 응답 저장
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                except Exception as e:
                    message_placeholder.error(f"오류가 발생했습니다: {e}")

# 초기 안내 메시지
if not st.session_state.messages:
    st.info("👈 사이드바에서 '챗봇 초기화' 버튼을 클릭하여 시작하세요.") 