import streamlit as st  # 스트림릿 라이브러리   
import os  # 파일 경로 처리
import pickle  # 파일 저장 및 로드
from langchain.chains import ConversationalRetrievalChain  # 대화형 검색 체인
from langchain.chat_models import ChatOpenAI  # 챗봇 모델
from langchain.memory import ConversationBufferMemory  # 대화 기록 메모리
from langchain.vectorstores import SKLearnVectorStore  # 벡터 저장소
from langchain.document_loaders import PyPDFLoader  # PDF 파일 로드
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI 임베딩 모델
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

# Streamlit Secrets에서 OpenAI API 키 가져오기
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key and hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

# 페이지 설정
st.set_page_config(
    page_title="현대자동차 아반떼 2025 설명서 챗봇",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 추가 - 간소화된 스타일
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
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
    button {
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# 벡터 저장소 생성 함수
def create_vectorstore():
    """
    PDF 문서를 로드하고, 청크로 분할한 후 벡터 저장소를 생성하는 함수
    
    Returns:
        SKLearnVectorStore: 생성된 벡터 저장소
    """
    try:
        # PDF 파일 경로 설정 (data 폴더 내의 모든 PDF 파일)
        pdf_folder_path = "./data/"
        
        # data 폴더가 없으면 생성
        if not os.path.exists(pdf_folder_path):
            os.makedirs(pdf_folder_path, exist_ok=True)
            st.warning(f"'{pdf_folder_path}' 폴더가 생성되었습니다. PDF 파일을 업로드해주세요.")
            return None
        
        # PDF 파일 목록 가져오기
        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.warning(f"'{pdf_folder_path}' 폴더에 PDF 파일이 없습니다. PDF 파일을 업로드해주세요.")
            # 디버깅 정보 출력
            st.info(f"현재 작업 디렉토리: {os.getcwd()}")
            st.info(f"data 폴더 경로: {os.path.abspath(pdf_folder_path)}")
            st.info(f"data 폴더 내 파일 목록: {os.listdir(pdf_folder_path) if os.path.exists(pdf_folder_path) else '폴더가 존재하지 않음'}")
            return None
        
        # 모든 문서를 저장할 리스트
        all_docs = []
        
        # 각 PDF 파일 처리
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder_path, pdf_file)
            st.info(f"'{pdf_file}' 파일을 처리 중입니다...")
            
            # PDF 로더를 사용하여 문서 로드
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 문서를 all_docs에 추가
            all_docs.extend(documents)
        
        # 문서를 청크로 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 각 청크의 최대 문자 수
            chunk_overlap=200,  # 청크 간 중복되는 문자 수
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(all_docs)
        st.info(f"총 {len(chunks)}개의 청크로 분할되었습니다.")
        
        # OpenAI API 키 확인
        if not openai_api_key:
            st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets 설정에서 OPENAI_API_KEY를 추가해주세요.")
            return None
        
        # OpenAI 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # scikit-learn 벡터 저장소 생성
        vectorstore = SKLearnVectorStore.from_documents(chunks, embeddings)
        
        # sklearn_index 폴더가 없으면 생성
        if not os.path.exists("sklearn_index"):
            os.makedirs("sklearn_index", exist_ok=True)
        
        # 벡터 저장소 저장 (pickle 사용)
        vectorstore_path = "sklearn_index/vectorstore.pkl"
        with open(vectorstore_path, "wb") as f:
            pickle.dump(vectorstore, f)
        
        if os.path.exists(vectorstore_path):
            st.success(f"벡터 저장소가 '{vectorstore_path}' 파일에 저장되었습니다.")
        else:
            st.error(f"벡터 저장소 저장에 실패했습니다. 경로: {vectorstore_path}")
        
        return vectorstore
    
    except Exception as e:
        st.error(f"벡터 저장소 생성 중 오류가 발생했습니다: {str(e)}")
        # 디버깅을 위한 정보 출력
        st.info(f"현재 작업 디렉토리: {os.getcwd()}")
        return None

# 벡터 저장소 로드 함수
@st.cache_resource
def load_vectorstore():
    """
    저장된 벡터 저장소를 로드하는 함수
    
    Returns:
        SKLearnVectorStore: 로드된 벡터 저장소
    """
    try:
        vectorstore_path = "sklearn_index/vectorstore.pkl"
        
        # 벡터 저장소 파일이 존재하는지 확인
        if not os.path.exists(vectorstore_path):
            st.warning(f"벡터 저장소 파일이 존재하지 않습니다: {vectorstore_path}")
            st.info("PDF 파일을 업로드하고 '벡터 저장소 생성' 버튼을 클릭하여 벡터 저장소를 생성해주세요.")
            # 디버깅 정보 출력
            st.info(f"현재 작업 디렉토리: {os.getcwd()}")
            st.info(f"sklearn_index 폴더 존재 여부: {os.path.exists('sklearn_index')}")
            if os.path.exists('sklearn_index'):
                st.info(f"sklearn_index 폴더 내 파일 목록: {os.listdir('sklearn_index')}")
            return None
        
        # 벡터 저장소 로드
        with open(vectorstore_path, "rb") as f:
            vectorstore = pickle.load(f)
        
        st.success("벡터 저장소를 성공적으로 로드했습니다.")
        return vectorstore
    
    except Exception as e:
        st.error(f"벡터 저장소 로드 중 오류가 발생했습니다: {str(e)}")
        # 디버깅을 위한 정보 출력
        st.info(f"현재 작업 디렉토리: {os.getcwd()}")
        return None

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
    
    # OpenAI API 키 확인
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Cloud의 Secrets 설정에서 OPENAI_API_KEY를 추가해주세요.")
        return None
    
    # ChatOpenAI 모델 초기화
    llm = ChatOpenAI(
        model_name="gpt-4o",  # gpt-3.5-turbo에서 gpt-4o로 변경
        temperature=0.2,  # 응답의 창의성 정도 (0에 가까울수록 결정적인 응답)
        openai_api_key=openai_api_key
    )
    
    # 대화형 검색 체인 생성
    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return chatbot

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'ready' not in st.session_state:
    st.session_state.ready = False

# 사이드바 구성
with st.sidebar:
    st.header("챗봇 설정")
    
    # 안내 메시지 추가
    st.info("PDF 파일을 업로드하면 자동으로 벡터 저장소가 생성되고 챗봇이 초기화됩니다.")
    
    # 수동 초기화 버튼 (필요한 경우에만 사용)
    with st.expander("고급 설정", expanded=False):
        st.caption("아래 버튼은 필요한 경우에만 사용하세요.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("챗봇 초기화", use_container_width=True):
                with st.spinner("챗봇을 초기화 중입니다..."):
                    vectorstore = load_vectorstore()
                    if vectorstore:
                        st.session_state.chatbot = create_chatbot()
                        st.session_state.ready = True
                        st.success("챗봇이 준비되었습니다!")
                        st.experimental_rerun()
        
        with col2:
            if st.button("벡터 저장소 생성", use_container_width=True):
                with st.spinner("벡터 저장소를 생성 중입니다..."):
                    vectorstore = create_vectorstore()
                    if vectorstore:
                        st.success("벡터 저장소가 성공적으로 생성되었습니다!")
                        st.session_state.ready = False
                        st.info("이제 '챗봇 초기화' 버튼을 클릭하여 챗봇을 초기화해주세요.")
    
    # PDF 파일 업로드 기능
    st.subheader("PDF 파일 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    
    if uploaded_file is not None:
        try:
            # data 폴더가 없으면 생성
            if not os.path.exists("./data/"):
                os.makedirs("./data/", exist_ok=True)
            
            # 업로드된 파일 저장
            file_path = os.path.join("./data/", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if os.path.exists(file_path):
                st.success(f"'{uploaded_file.name}' 파일이 성공적으로 업로드되었습니다!")
                
                # 파일 업로드 후 자동으로 벡터 저장소 생성
                with st.spinner("벡터 저장소를 자동으로 생성 중입니다..."):
                    vectorstore = create_vectorstore()
                    if vectorstore:
                        st.success("벡터 저장소가 성공적으로 생성되었습니다!")
                        
                        # 벡터 저장소 생성 후 자동으로 챗봇 초기화
                        with st.spinner("챗봇을 초기화 중입니다..."):
                            st.session_state.chatbot = create_chatbot()
                            st.session_state.ready = True
                            st.success("챗봇이 준비되었습니다! 이제 질문을 입력하세요.")
                            st.experimental_rerun()
            else:
                st.error(f"파일 저장에 실패했습니다. 경로: {file_path}")
        except Exception as e:
            st.error(f"파일 업로드 중 오류가 발생했습니다: {str(e)}")
            # 디버깅을 위한 정보 출력
            st.info(f"현재 작업 디렉토리: {os.getcwd()}")
            st.info(f"파일 이름: {uploaded_file.name}")
            st.info(f"파일 크기: {len(uploaded_file.getbuffer())} 바이트")
    
    st.markdown("---")
    st.markdown("### 예시 질문")
    example_questions = [
        "아반떼 엔진 오일은 어떻게 교체하나요?",
        "타이어 공기압은 얼마로 유지해야 하나요?",
        "타이어가 펑크났어. 해결책을 알려줘",
        "창문에 서리가 자꾸 껴요. 어떻게 해야 하나요?",
        "연비를 향상시키는 방법이 있을까요?"
    ]
    
    for q in example_questions:
        if st.button(q, use_container_width=True):
            if st.session_state.ready:
                st.session_state.messages.append({"role": "user", "content": q})
                st.experimental_rerun()
            else:
                st.warning("먼저 챗봇을 초기화해주세요.")

# 메인 영역 구성
# 제목 및 소개
st.title("🚗 현대자동차 설명서 챗봇")
st.markdown("""
이 챗봇은 현대자동차 아반떼 2025 모델에 대한 정보를 제공합니다.
RAG(Retrieval-Augmented Generation) 기술을 활용하여 PDF 형식의 설명서에서 관련 정보를 검색하고,
이를 기반으로 정확한 답변을 생성합니다.

**사용 방법:**
1. 사이드바에서 PDF 형식의 설명서 파일을 업로드하세요.
2. 파일 업로드 후 자동으로 벡터 저장소가 생성되고 챗봇이 초기화됩니다.
3. 아래 입력창에 질문을 입력하면 답변을 받을 수 있습니다.
""")

# 챗봇 생성
if 'chatbot' not in st.session_state:
    chatbot = create_chatbot()
    if chatbot:
        st.session_state.chatbot = chatbot
        st.session_state.ready = True
else:
    chatbot = st.session_state.chatbot

# 구분선 추가
st.markdown("---")

# 채팅 영역
if not st.session_state.ready:
    st.info("PDF 파일을 업로드하면 자동으로 챗봇이 초기화됩니다. 사이드바에서 PDF 파일을 업로드해주세요.")
    # 화살표로 사이드바 방향 표시
    st.markdown("👈 왼쪽 사이드바에서 PDF 파일을 업로드하세요.")

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
    
    if st.session_state.ready:
        with st.spinner("답변을 생성 중입니다..."):
            # 챗봇에 질문하고 응답 받기
            response = st.session_state.chatbot({"question": prompt})
            answer = response["answer"]
            
            # 참고 페이지 추출
            if "source_documents" in response:
                pages = [doc.metadata.get('page', 'N/A') for doc in response["source_documents"]]
                answer += f"\n\n**참고 페이지**: {', '.join(map(str, pages))}"
            
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
        st.error("챗봇이 초기화되지 않았습니다. 사이드바에서 '챗봇 초기화' 버튼을 클릭하여 시작하세요.")

# 푸터
st.markdown("---")
st.markdown("© 현대자동차 아반떼 2025 설명서 챗봇 | 개발: Daniel8824") 