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

from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
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

    return STEmbedding("dragonkue/snowflake-arctic-embed-l-v2.0-ko")

# 벡터스토어 생성
@st.cache_resource
def create_vector_store(file_path: str, cache_buster: str):
    docs = load_csv_and_create_docs(file_path, cache_buster)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    file_hash = os.path.splitext(os.path.basename(file_path))[0]
    collection_name = f"coll_{file_hash}"

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
        persist_directory=None,
    )
    return vectorstore, split_docs  # split_docs도 함께 반환

# BM25 용 한국어 토크나이저
@st.cache_resource
def get_kiwi():
    return Kiwi()

kiwi = get_kiwi()

# Kiwi로 형태소만 추출하는 함수
def tokenize(text):
    # 첫 번째 분석 결과에서 형태소만 추출
    return [morph for morph, pos, start, length in result if pos.startswith(("NN", "VV", "VA"))]

# RAG 체인 초기화
@st.cache_resource
def initialize_components(file_path: str, selected_model: str, cache_buster: str):
    vectorstore, split_docs = create_vector_store(file_path, cache_buster)

    # BM25Retriever 생성 (원문 유지 + tokenizer 지정)
    bm25_retriever = BM25Retriever.from_documents(
        documents=split_docs,         # Document 객체 리스트를 직접 전달
        preprocess_func=tokenize
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
                if score > 0.01:
                    doc.metadata["relevance_score"] = score
                    result.append(doc)
            return result

    @st.cache_resource
    def get_cross_encoder():
        return HuggingFaceCrossEncoder(model_name="dragonkue/bge-reranker-v2-m3-ko")
    
    model = get_cross_encoder()
    compressor = CrossEncoderRerankerWithScore(model=model, top_n=30)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습니다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        ("human", "{input}"),
    ])
    llm = ChatOpenAI(model=selected_model)

    # retriever가 바로 문서 내용을 {context}에 채워주는 역할을 합니다.
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # [수정] history_aware_retriever 대신 준비된 compression_retriever를 직접 연결합니다.
    rag_chain = create_retrieval_chain(compression_retriever, question_answer_chain)

    return rag_chain



def main():

    st.title("🧠 RAG 질의응답")
    st.subheader("설문 응답을 의미 단위로 분리한뒤(semantic chuncking) 키워드를 도출하고, 이를 활용하여 분석을 진행합니다.")
    st.text("예시) 유저A: '그래픽은 좋지만 사운드는 별로입니다' -> 유저A는 '그래픽은 좋다' 와 '사운드는 별로다' 두 가지 주제를 얘기하고 있습니다. LLM을 활용하여 이를 의미 단위로 분리(이하 '청크')하는 전처리를 진행하였습니다.")
    st.markdown("---")

    # 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요. 새롭게 파일을 넣는 경우, 좌측 상단 새로고침 버튼을 누르세요", 
        type=['csv'],
        help="user_id, total_cl, name, keywords, summary, SPLITTED 컬럼 필요"
    )

    if uploaded_file is not None:
        try:
            # CSV 파일 읽기
            df = pd.read_csv(uploaded_file)

            # 컬럼 확인 (name 컬럼 추가)
            mindmap_columns = ['user_id', 'total_cl', 'name', 'keywords', 'summary', 'SPLITTED']
            has_mindmap_columns = all(col in df.columns for col in mindmap_columns)

            if not has_mindmap_columns:
                st.error("마인드맵 또는 RAG 기능을 위한 필수 컬럼이 없습니다.")
                st.info("user_id, total_cl, name, keywords, summary, SPLITTED")
                st.stop()

            
            st.subheader("📊 데이터 요약")
            
            if has_mindmap_columns:
                # 기본 정보 메트릭
                filtered_df = df[df.total_cl != 99]
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("전체 응답 수", df.user_id.nunique())
                with col2:
                    st.metric("전체 청크 수", len(df))

            if has_mindmap_columns:
                # Summary Table (4단계 구조)
                st.subheader("📋 키워드 미분류 청크")
                st.text("키워드로 분류되지 않은 청크들을 확인할 수 있습니다.(테이블 우측 상단 다운로드 가능)")
                no_filtered_df = df[["user_id","SPLITTED"]]
                st.dataframe(
                    no_filtered_df.set_index("user_id"),
                    use_container_width=True,
                )

            st.subheader("🤖 RAG 질의응답")
            st.text("청크를 근거로 유저의 질의에 응답하며, 응답에 사용된 청크를 확인할 수 있습니다.(현재 상위 10개만 확인 가능)")
            st.markdown("RAG 구축 간 시간이 소요됩니다.(약 N분)")
            
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

            # 채팅 초기화/새 세션 시작 버튼
            btn_col1, btn_col2 = st.columns([1, 1])
            with btn_col1:
                if st.button("채팅 히스토리 지우기", use_container_width=True):
                    chat_history.clear()  # 현재 세션의 메시지 비움
                    st.rerun()
            
            with btn_col2:
                if st.button("새 채팅 시작", use_container_width=True):
                    st.session_state["chat_session_nonce"] += 1  # 새 세션
                    # 메모리에 남아있는 현재 키 정리(선택)
                    if chat_history_key in st.session_state:
                        del st.session_state[chat_history_key]
                    st.rerun()

            if len(chat_history.messages) == 0:
                chat_history.add_ai_message("업로드된 유저 응답 기반으로 무엇이든 물어보세요! 🤗")

            for msg in chat_history.messages:
                st.chat_message(msg.type).write(msg.content)

            if prompt_message := st.chat_input("질문을 입력하세요"):
                st.chat_message("human").write(prompt_message)
                with st.chat_message("ai"):
                    with st.spinner("생각 중입니다..."):

                        response = conversational_rag_chain.invoke(
                            {"input": prompt_message},
                            config,
                        )
                        answer = response['answer']
                        st.write(answer)

                        if "관련된 내용이 없습니다" not in answer and response.get("context"):
                            with st.expander("참고 문서 확인"):
                                seen = set()
                                for doc in response['context']:
                                    key = (doc.metadata.get("source"), doc.page_content)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                
                                    source = doc.metadata.get('source', '알 수 없음')
                                    raw_ans = doc.metadata.get('ans', '알 수 없음')
                                    #score = doc.metadata.get('score', None)
                                    source_filename = os.path.basename(source)
                                
                                    st.markdown(f"👤 {source_filename}")
                                    st.html(raw_ans)

        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
            st.exception(e)

    else:
        # 샘플 정보 표시
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("💡 CSV 파일을 업로드하면 4단계 하이브리드 마인드맵과 RAG 챗봇을 사용할 수 있습니다.")

            with st.expander("🌳 4단계 하이브리드 마인드맵의 특징"):
                st.markdown("""
                **🏗️ 4단계 트리 + Force 구조**
                - 메인 주제가 왼쪽에 위치
                - 상위개념(name)이 첫 번째 확장
                - 키워드들이 두 번째 확장 
                - 요약들이 세 번째 확장
                - Force Simulation으로 겹침 방지
                
                **🎯 인터랙션**  
                - 메인 주제 클릭 → 모든 상위개념 표시
                - 상위개념 클릭 → 해당 키워드들 표시
                - 키워드 클릭 → 해당 요약들 표시
                - 드래그로 노드 자유 이동
                - 트리 복원으로 언제든 원래 형태 복귀
                - 물리엔진 토글로 겹침 방지 제어
                """)

        with col2:
            with st.expander("📋 CSV 파일 형식 요구사항 (4단계)"):
                st.markdown("""
                **마인드맵용 (필수):**
                ```
                user_id, total_cl, name, keywords, summary
                user001, 1, "제품품질", "품질", "제품이 만족스럽다"
                user002, 2, "가격정책", "가격", "가격이 합리적이다"
                user003, 99, "", "", "무효 응답"
                ```
                
                **RAG 챗봇용 (추가 필요):**
                ```
                SPLITTED, highlighted_ans
                "제품에 대한 상세한 의견...", "원본 응답..."
                "서비스 경험에 대한 설명...", "원본 응답..."
                ```
                
                **4단계 구조**: 메인 → 상위개념(name) → 키워드 → 요약
                * total_cl != 99 인 데이터만 마인드맵에 사용됩니다
                * 모든 기능을 사용하려면 모든 컬럼이 필요합니다
                """)

if st.button("🔄 새로고침 버튼을 누르세요"):
    st.cache_resource.clear()
    shutil.rmtree(os.path.join(tempfile.gettempdir(), "chroma_db_user"), ignore_errors=True)
    st.success("초기화 완료")
    st.rerun()

if __name__ == "__main__":
    main()
