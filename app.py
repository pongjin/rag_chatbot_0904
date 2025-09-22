import os
import tempfile
import hashlib
import shutil
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np
import sys
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# RAG 관련 imports
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableMap
from langchain_core.embeddings import Embeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FakeEmbeddings
from typing import List, Sequence
from sentence_transformers import SentenceTransformer

import hashlib
import shutil

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain.schema import Document
from langchain_core.runnables import Runnable

from kiwipiepy import Kiwi

# 파일 해시 생성
def get_file_hash(uploaded_file):
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# pysqlite3 패치
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma   # ✅ Chroma import
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']


st.set_page_config(page_title="RAG Chatbot", page_icon="🧠", layout="wide")

# CSV 로딩 → 유저 단위로 문서 생성
@st.cache_resource
def load_csv_and_create_docs(file_path: str, cache_buster: str):
    df = pd.read_csv(file_path)

    if 'user_id' not in df.columns or 'SPLITTED' not in df.columns or 'highlighted_ans' not in df.columns:
        st.error("해당하는 컬럼 없음")
        return []

    docs = []
    for idx, row in df.iterrows():
        content = str(row['SPLITTED'])  # 한 행의 SPLITTED 값
        metadata = {
                "source": f"row_{idx}",
                "ans": str(row['highlighted_ans']),
                   }  # 행 인덱스를 소스로 사용
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

@st.cache_resource
def get_embedder():
    class STEmbedding(Embeddings):
        def __init__(self, model_name: str):
            # 안내 메시지 출력
            with st.spinner(f"임베딩 모델을 로드하는 중입니다... ({model_name})"):
                try:
                    self.model = SentenceTransformer(model_name)
                    st.success(f"임베딩 모델 로드 성공: {model_name}")
                except Exception as e:
                    st.error(f"임베딩 모델 로드 실패: {str(e)}")
                    raise e

        def embed_documents(self, texts):
            # 리스트 입력에 대해 배치 인코딩
            return self.model.encode(list(texts), normalize_embeddings=True).tolist()

        def embed_query(self, text):
            # 단일 쿼리 인코딩
            return self.model.encode(text, normalize_embeddings=True).tolist()

    return STEmbedding("dragonkue/snowflake-arctic-embed-l-v2.0-ko") #sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 벡터스토어 생성
@st.cache_resource
def create_vector_store(file_path: str, cache_buster: str):
    docs = load_csv_and_create_docs(file_path, cache_buster)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    file_hash = os.path.splitext(os.path.basename(file_path))[0]
    collection_name = f"coll_{file_hash}_{cache_buster}"
    #collection_name = f"coll_{file_hash}"

    # 쓰기 가능한 루트 (예: /tmp)
    persist_root = os.path.join(tempfile.gettempdir(), "chroma_db_user")
    persist_dir = os.path.join(persist_root, collection_name)

    # 폴더 깨끗하게 재생성
    shutil.rmtree(persist_dir, ignore_errors=True)
    os.makedirs(persist_dir, exist_ok=True)

    embeddings = get_embedder()
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir, #None,
    )
    return vectorstore, split_docs  # split_docs도 함께 반환


# BM25 용 한국어 토크나이저
'''
@st.cache_resource
def get_kiwi():
    return Kiwi()

kiwi = get_kiwi()

# Kiwi로 형태소만 추출하는 함수
def tokenize(text):
    # 첫 번째 분석 결과에서 형태소만 추출
    result = kiwi.analyze(text)[0][0]
    return [morph for morph, pos, start, length in result if pos.startswith(("NN", "VV", "VA"))]
'''

# RAG 체인 초기화
@st.cache_resource
def initialize_components(file_path: str, selected_model: str, cache_buster: str):
    vectorstore, split_docs = create_vector_store(file_path, cache_buster)

    # BM25Retriever 생성 (원문 유지 + tokenizer 지정)
    bm25_retriever = BM25Retriever.from_documents(
        documents=split_docs,         # Document 객체 리스트를 직접 전달
        #preprocess_func=tokenize
    )
    bm25_retriever.k = 15  # BM25Retriever의 검색 결과 개수를 20으로 설정

    # Chroma retriever 생성
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    
    # 앙상블 retriever 초기화
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.2, 0.8],  # BM25: 20%, Chroma: 80%
    )

    # --- 위에서 정의한 커스텀 클래스 ---
    class CrossEncoderRerankerWithScore(CrossEncoderReranker):
        """점수를 메타데이터에 추가하는 CrossEncoderReranker"""
        def compress_documents(
            self, documents: Sequence[Document], query: str, callbacks=None
        ) -> Sequence[Document]:
            if not documents: return []
            doc_list = [doc.page_content for doc in documents]
            _scores = self.model.score(list(zip([query] * len(doc_list), doc_list)))
            docs_with_scores = sorted(zip(documents, _scores), key=lambda x: x[1], reverse=True)

            result = []
            for doc, score in docs_with_scores[: self.top_n]:
                # 👇 [수정] 점수가 0.0010을 넘는 문서만 결과에 추가하도록 수정
                if score > 0.0:
                    doc.metadata["relevance_score"] = score
                    result.append(doc)
            return result
    '''
    @st.cache_resource
    def get_cross_encoder():
        return HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker") #  dragonkue/bge-reranker-v2-m3-ko
    
    model = get_cross_encoder()
    compressor = CrossEncoderRerankerWithScore(model=model, top_n=30)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    '''
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습니다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=selected_model)

    # retriever가 바로 문서 내용을 {context}에 채워주는 역할을 합니다.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # [수정] history_aware_retriever 대신 준비된 compression_retriever를 직접 연결합니다.
    #rag_chain = create_retrieval_chain(compression_retriever , question_answer_chain)
    rag_chain = create_retrieval_chain(ensemble_retriever , question_answer_chain)

    return rag_chain



def main():

    st.title("🧠 주관식 데이터 검색기")
    st.subheader("설문 응답을 의미 단위로 분리한뒤(semantic chuncking) 키워드를 도출하고, 이를 활용하여 분석을 진행합니다.")
    st.text("예시) 유저A: '그래픽은 좋지만 사운드는 별로입니다' -> 유저A는 '그래픽은 좋다' 와 '사운드는 별로다' 두 가지 주제를 얘기하고 있습니다. LLM을 활용하여 이를 의미 단위로 분리(이하 '청크')하는 전처리를 진행하였습니다.")
    st.markdown("---")

    # 파일 업로드
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0
    
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요. 새롭게 파일을 넣는 경우, 좌측 상단 새로고침 버튼을 누르세요", 
        type=['csv'],
        help="user_id, SPLITTED, highlighted_ans 컬럼 필요",
        key=f"file_uploader_{st.session_state['uploader_key']}"  # ✅ 세션 키 적용
    )

    if uploaded_file is not None:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(uploaded_file)

            # 컬럼 확인 (name 컬럼 추가)
            mindmap_columns = ['user_id','highlighted_ans', 'SPLITTED']
            has_mindmap_columns = all(col in df.columns for col in mindmap_columns)

            if not has_mindmap_columns:
                st.error("마인드맵 또는 RAG 기능을 위한 필수 컬럼이 없습니다.")
                st.info("user_id, SPLITTED, highlighted_ans")
                st.stop()

            
            st.subheader("📊 데이터 요약")
            
            if has_mindmap_columns:
                # 기본 정보 메트릭
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("전체 응답 수(불성실 제외)", df[df.keyword != '없음'].user_id.nunique())
                with col2:
                    st.metric("전체 청크 수(불성실 제외)", len(df[df.keyword != '없음']))
                with col3:
                    # 상위 10개 키워드 추출
                    df['clean_keyword'] = df['keyword'].apply(lambda x: x.replace(' ',''))
                    df_cnt = pd.DataFrame(df.groupby('clean_keyword').user_id.nunique().sort_values(ascending= False)).reset_index()
                    top10 = df_cnt[df_cnt.clean_keyword != '없음'].head(10)
                    # Noto Sans KR (TTF 버전) 다운로드
                    url = "https://github.com/moonspam/NanumSquare/raw/master/NanumSquareR.ttf"
                    font_path = "NanumSquare.ttf"
                    
                    if not os.path.exists(font_path):
                        r = requests.get(url)
                        with open(font_path, "wb") as f:
                            f.write(r.content)
                    
                    # 파일 크기 확인 (정상적으로 받았는지 체크)
                    st.text("주로 언급된 키워드")
                    
                    wc = WordCloud(
                        font_path=font_path, 
                        background_color="white", 
                        width=200, 
                        height=100
                    ).generate_from_frequencies(dict(zip(top10['clean_keyword'], top10['user_id'])))
                    
                    # 시각화
                    fig, ax = plt.subplots(figsize=(2, 1))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig, use_container_width=False)

            if has_mindmap_columns:

                st.subheader("📋 전체 청크")
                st.text("불성실 응답을 제외한 전체 청크들을 확인할 수 있습니다.(테이블 우측 상단 내 검색 및 다운로드 가능)")
                no_filtered_df = df[df.SPLITTED != '없음'][["user_id","SPLITTED"]]
                st.dataframe(
                    no_filtered_df.set_index("user_id"),
                    use_container_width=True,
                )

            
            st.subheader("🤖 RAG 질의응답")
            st.text("청크를 근거로 유저의 질의에 응답하며, 응답에 사용된 청크를 확인할 수 있습니다.(최대 30개 까지 확인 가능)")
            st.markdown("청크 크기에 따라 RAG 구축 시간이 소요됩니다.(약 N분)")
            
            file_hash = get_file_hash(uploaded_file)

            # 세션 상태 초기화
            if "chat_session_nonce" not in st.session_state:
                st.session_state["chat_session_nonce"] = 0
            
            # 파일이 바뀌면 히스토리 초기화
            if st.session_state.get("last_file_hash") != file_hash:
                # 기존 히스토리 키가 있으면 제거
                old_key = st.session_state.get("chat_history_key")
                if old_key and old_key in st.session_state:
                    del st.session_state[old_key]
                st.session_state["last_file_hash"] = file_hash
                st.session_state["chat_session_nonce"] = 0  # 파일 바뀌면 nonce 초기화
            
            # 현재 세션 식별자(파일 해시 + nonce)
            chat_session_id = f"{file_hash}-{st.session_state['chat_session_nonce']}"
            chat_history_key = f"chat_messages_{chat_session_id}"
            
            # 이 값을 저장해두면 다음 턴에서 접근 가능
            st.session_state["chat_history_key"] = chat_history_key
            
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"{file_hash}.csv")

            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            rag_chain = initialize_components(temp_path, "gpt-4o-mini", cache_buster=file_hash)
            chat_history = StreamlitChatMessageHistory(key=chat_history_key)
            config = {"configurable": {"session_id": chat_session_id}}

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                lambda session_id: chat_history,
                input_messages_key="input",
                history_messages_key="history",
                output_messages_key="answer",
            )

            if st.button("채팅 히스토리 지우기", use_container_width=True):
                chat_history.clear()  # 현재 세션의 메시지 비움
                st.rerun()

            if len(chat_history.messages) == 0:
                chat_history.add_ai_message("업로드된 유저 응답 기반으로 무엇이든 물어보세요! 🤗")

            # 히스토리 출력
            for msg in chat_history.messages:
                if msg.type == "human":
                    st.chat_message("human").write(msg.content)
                elif msg.type == "ai":
                    try:
                        content = json.loads(msg.content)
                        st.chat_message("ai").write(content["answer"])
            
                        if content.get("context"):
                            with st.expander("참고 문서 확인", expanded=False):
                                seen = set()
                                for doc in content["context"]:
                                    key = (doc["source"], doc["page_content"])
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    st.markdown(f"👤 {doc['source']}")
                                    st.html(doc["ans"])
                    except json.JSONDecodeError:
                        st.chat_message("ai").write(msg.content)
            
            if prompt_message := st.chat_input("질문을 입력하세요"):
                #st.chat_message("human").write(prompt_message)
                #with st.chat_message("ai"):
                with st.spinner(f"{prompt_message} 응답 생성중..."):

                    # 사용자 메시지 먼저 추가
                    chat_history.add_user_message(prompt_message)
                    
                    # 기본 rag_chain 사용 (자동 히스토리 저장 없음)
                    response = rag_chain.invoke({"input": prompt_message, "history": chat_history.messages})
                    
                    answer = response['answer']

                    '''
                    response = conversational_rag_chain.invoke(
                        {"input": prompt_message},
                        config,
                    )
                    answer = response['answer']
                    '''
        
                    # Document 객체를 직렬화 가능한 dict로 변환
                    context = []
                    if "관련된 내용이 없습니다" not in answer and response.get("context"):
                        for doc in response["context"]:
                            context.append({
                                "source": doc.metadata.get("source", "알 수 없음"),
                                "ans": doc.metadata.get("ans", "알 수 없음"),
                                "page_content": doc.page_content
                            })
        
                    # JSON으로 직렬화해서 저장
                    chat_history.add_ai_message(
                        json.dumps({"answer": answer, "context": context}, ensure_ascii=False)
                    )
                    
                    st.rerun()

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.exception(e)

    else:
        # 샘플 정보 표시
        col1, col2 = st.columns([1, 1])

if st.button("🔄 새로고침 (모든 기록 삭제)"):
    # 1. Streamlit의 리소스 캐시 초기화
    st.cache_resource.clear()

    # 2. 디스크에 저장된 ChromaDB 파일 삭제
    # 경로가 존재하는지 확인 후 삭제하는 것이 더 안전합니다.
    chroma_db_path = os.path.join(tempfile.gettempdir(), "chroma_db_user")
    if os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path, ignore_errors=True)

    # 3. 세션 상태(채팅 기록 등) 완전 초기화
    # st.session_state의 모든 키를 순회하며 삭제합니다.
    for key in list(st.session_state.keys()):
        del st.session_state[key]

    # 4. 파일 업로더 키 갱신 → 업로드 표시 지움
    st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1  
    
    st.success("✅ 모든 캐시와 채팅 기록이 초기화되었습니다!")
    st.rerun()

if __name__ == "__main__":
    main()
