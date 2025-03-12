import streamlit as st  # 스트림릿 라이브러리   
import os  # 파일 경로 처리
import pickle  # 파일 저장 및 로드
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.chat_models import ChatOpenAI  # 챗봇 모델
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
from langchain.vectorstores import SKLearnVectorStore  # 벡터 저장소
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="현대자동차 챗봇",
    page_icon="🚗",
    layout="centered"
)

# 스타일 추가
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .chat-message.user {
        background-color: #e6f7ff;
        border-left: 5px solid #1890ff;
    }
    .chat-message.bot {
        background-color: #f6ffed;
        border-left: 5px solid #52c41a;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# 제목 및 소개
st.title("🚗 현대자동차 챗봇")
st.markdown("""
이 챗봇은 현대자동차 아반떼 2025 모델에 대한 정보를 제공합니다.
차량의 기능, 사양, 유지 관리 등에 대해 질문해보세요!
""")

# 벡터 저장소 로드 함수
@st.cache_resource
def load_vectorstore():
    """
    저장된 벡터 저장소를 로드하는 함수
    
    Returns:
        SKLearnVectorStore: 로드된 벡터 저장소
    """
    # 벡터 저장소 파일 경로 확인
    vectorstore_path = "sklearn_index/vectorstore.pkl"
    
    if not os.path.exists(vectorstore_path):
        st.error("벡터 저장소가 존재하지 않습니다. 먼저 create_vectorstore.py를 실행하여 벡터 저장소를 생성해주세요.")
        return None
    
    # 벡터 저장소 로드 (pickle 사용)
    with open(vectorstore_path, "rb") as f:
        vectorstore = pickle.load(f)
    
    return vectorstore

# 챗봇 생성 함수
@st.cache_resource
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
        model_name="gpt-4o",  # gpt-3.5-turbo에서 gpt-4o로 변경
        temperature=0.2  # 응답의 창의성 정도 (0에 가까울수록 결정적인 응답)
    )
    
    # 대화형 검색 체인 생성
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return chatbot

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 챗봇 생성
chatbot = create_chatbot()

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.container():
        st.markdown(f"""
        <div class="chat-message {'bot' if message['role'] == 'assistant' else 'user'}">
            <div class="avatar">
                {'🤖' if message['role'] == 'assistant' else '👤'}
            </div>
            <div class="message">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 사용자 메시지 표시
    with st.container():
        st.markdown(f"""
        <div class="chat-message user">
            <div class="avatar">👤</div>
            <div class="message">{prompt}</div>
        </div>
        """, unsafe_allow_html=True)
    
    if chatbot:
        with st.spinner("답변을 생성 중입니다..."):
            # 챗봇에 질문하고 응답 받기
            response = chatbot({"question": prompt})
            answer = response["answer"]
            
            # 챗봇 메시지 추가
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
            # 챗봇 메시지 표시
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">{answer}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error("챗봇이 초기화되지 않았습니다. 벡터 저장소가 생성되었는지 확인해주세요.")

# 푸터
st.markdown("---")
st.markdown("© 2023 현대자동차 챗봇 | 개발: AI 어시스턴트") 