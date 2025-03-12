# 현대자동차 설명서 챗봇

이 프로젝트는 현대자동차 설명서를 기반으로 사용자의 질문에 답변하는 챗봇을 구현합니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여 PDF 형식의 설명서에서 관련 정보를 검색하고, 이를 기반으로 정확한 답변을 생성합니다.

## 기능

- PDF 형식의 현대자동차 설명서 처리
- 문서를 청크로 분할하여 벡터 저장소 생성 (scikit-learn 사용)
- 사용자 질문에 대한 관련 정보 검색
- 검색된 정보를 기반으로 자연스러운 답변 생성 (GPT-4o 모델 사용)
- 대화 기록 유지를 통한 맥락 이해
- Streamlit을 활용한 웹 인터페이스 제공

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. OpenAI API 키 설정:
`.env` 파일에 OpenAI API 키를 입력합니다.
```
OPENAI_API_KEY=your_openai_api_key_here
```

## 사용 방법

### 로컬에서 실행하기

1. 현대자동차 설명서 PDF 파일을 `data` 폴더에 넣습니다.
   - 폴더가 없는 경우 자동으로 생성됩니다.

2. 벡터 저장소 생성:
```bash
python create_vectorstore.py
```

3. 챗봇 실행 (터미널 버전):
```bash
python hyundai_chatbot.py
```

4. 또는 Streamlit 웹 앱 실행:
```bash
streamlit run streamlit_app.py
```

5. 질문 입력:
   - 챗봇이 실행되면 현대자동차에 관한 질문을 입력할 수 있습니다.
   - 터미널 버전에서 종료하려면 'exit' 또는 '종료'를 입력하세요.

### Streamlit Cloud에 배포하기

1. GitHub 저장소에 코드 푸시

2. [Streamlit Cloud](https://streamlit.io/cloud)에 접속하여 로그인

3. "New app" 버튼 클릭

4. GitHub 저장소, 브랜치, 메인 파일(`streamlit_app.py`) 선택

5. 환경 변수 설정:
   - "Advanced settings" > "Secrets" 섹션에서 다음과 같이 TOML 형식으로 OpenAI API 키 추가
   ```
   OPENAI_API_KEY = "your_openai_api_key_here"
   ```

6. "Deploy!" 버튼 클릭

7. 배포 후 PDF 파일 업로드 및 벡터 저장소 생성:
   - 웹 앱의 사이드바에서 PDF 파일 업로드
   - "벡터 저장소 생성" 버튼 클릭
   - "챗봇 초기화" 버튼 클릭

## 웹 앱 사용 방법

1. 사이드바의 "PDF 파일 업로드" 섹션에서 현대자동차 설명서 PDF 파일 업로드

2. "벡터 저장소 생성" 버튼 클릭하여 벡터 저장소 생성

3. "챗봇 초기화" 버튼 클릭하여 챗봇 초기화

4. 예시 질문 버튼을 클릭하거나 직접 질문 입력

5. 챗봇의 답변 확인 (참고 페이지 정보 포함)

## 예시 질문

- "아반떼 엔진 오일은 어떻게 교체하나요?"
- "타이어 공기압은 얼마로 유지해야 하나요?"
- "타이어가 펑크났어. 해결책을 알려줘"
- "창문에 서리가 자꾸 껴요. 어떻게 해야 하나요?"
- "연비를 향상시키는 방법이 있을까요?"

## 주의사항

- 이 챗봇은 제공된 PDF 문서에 포함된 정보만을 기반으로 답변합니다.
- 정확한 답변을 위해 최신 현대자동차 설명서를 사용하세요.
- OpenAI API 사용에 따른 비용이 발생할 수 있습니다.
- GPT-4o 모델을 사용하므로 GPT-3.5-turbo보다 API 비용이 더 높을 수 있습니다.
- Streamlit Cloud에 배포할 경우, 무료 티어의 리소스 제한이 있을 수 있습니다.

## 기술 스택

- Python 3.8+
- LangChain: 대화형 검색 체인 구현
- OpenAI GPT-4o: 자연어 처리 및 응답 생성
- scikit-learn: 벡터 저장소 구현
- Streamlit: 웹 인터페이스 구현
- PyPDF: PDF 문서 처리 