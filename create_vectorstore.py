import os
from langchain.document_loaders import PyPDFLoader  # PDF 파일 로드
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 텍스트 분할
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAI 임베딩 모델
from langchain.vectorstores import SKLearnVectorStore  # scikit-learn 벡터 저장소
from dotenv import load_dotenv  # 환경 변수 로드

# .env 파일에서 환경 변수 로드
load_dotenv()

def create_vectorstore():
    """
    PDF 문서를 로드하고, 청크로 분할한 후 벡터 저장소를 생성하는 함수
    
    Returns:
        SKLearnVectorStore: 생성된 벡터 저장소
    """
    # PDF 파일 경로 설정 (data 폴더 내의 모든 PDF 파일)
    pdf_folder_path = "./data/"
    
    # data 폴더가 없으면 생성
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)
        print(f"'{pdf_folder_path}' 폴더가 생성되었습니다. PDF 파일을 이 폴더에 넣어주세요.")
        return None
    
    # PDF 파일 목록 가져오기
    pdf_files = [f for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"'{pdf_folder_path}' 폴더에 PDF 파일이 없습니다. PDF 파일을 추가해주세요.")
        return None
    
    # 모든 문서를 저장할 리스트
    all_docs = []
    
    # 각 PDF 파일 처리
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder_path, pdf_file)
        print(f"'{pdf_file}' 파일을 처리 중입니다...")
        
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
    print(f"총 {len(chunks)}개의 청크로 분할되었습니다.")
    
    # OpenAI 임베딩 모델 초기화
    embeddings = OpenAIEmbeddings()
    
    # scikit-learn 벡터 저장소 생성
    vectorstore = SKLearnVectorStore.from_documents(chunks, embeddings)
    
    # 벡터 저장소 저장
    vectorstore.save_local("sklearn_index")
    print("벡터 저장소가 'sklearn_index' 폴더에 저장되었습니다.")
    
    return vectorstore

if __name__ == "__main__":
    create_vectorstore() 