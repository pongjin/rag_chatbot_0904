import os
import tempfile
import hashlib
import shutil
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import numpy as np

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

from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

import hashlib
import shutil

# 파일 해시 생성
def get_file_hash(uploaded_file):
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()

# pysqlite3 패치
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

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
    return vectorstore

from langchain.schema import Document
from langchain_core.runnables import Runnable

class ScoredRetriever(Runnable):
    def __init__(self, vectorstore, k=10, score_threshold=0.1):
        self.vectorstore = vectorstore
        self.k = k
        self.score_threshold = score_threshold  # ✅ 임계값 추가

    def invoke(self, query, config=None):
        docs_and_scores = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=self.k
        )

        filtered_docs = []
        for doc, score in docs_and_scores:
            doc.metadata["score"] = score
            if score >= self.score_threshold:   # ✅ 0.1 이상만 남김
                filtered_docs.append(doc)

        return filtered_docs

# RAG 체인 초기화
@st.cache_resource
def initialize_components(file_path: str, selected_model: str, cache_buster: str):
    vectorstore = create_vector_store(file_path, cache_buster)

    # 기존 retriever 대신 ScoredRetriever 사용
    retriever = ScoredRetriever(vectorstore, k=10, score_threshold=0.1)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화 내용을 반영해 현재 질문을 독립형 질문으로 바꿔줘."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "다음 문서 내용을 참고하여 질문에 무조건 한국어로 답변해줘. 문서와 유사한 내용이 없으면 무조건 '관련된 내용이 없습니다'라고 말해줘. 꼭 이모지 써줘! 참고 문서는 아래에 보여줄 거야.\n\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model=selected_model)
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain

# 4단계 트리 데이터 생성 함수 (메인 → name → keywords → summary)
def create_tree_data_from_csv_4level(df):
    """
    CSV 데이터에서 4단계 트리 데이터 구조를 생성하는 함수
    구조: 메인 → 상위개념(name) → 키워드 → 요약
    """
    # summary_table 생성 (name, keywords, summary 기준으로 그룹화)
    summary_table = df[df.total_cl != 99].groupby(['name', 'keywords','summary'], as_index=False, dropna=False).agg({'user_id': 'nunique'}).rename(columns={'user_id': 'cnt'})

    # name별 총 cnt 계산
    name_totals = summary_table.groupby('name')['cnt'].sum().to_dict()
    
    # name-keywords 조합별 총 cnt 계산
    keyword_totals = summary_table.groupby(['name', 'keywords'])['cnt'].sum().to_dict()

    # 색상 팔레트 생성 (name별 메인 색상)
    name_colors_base = ['#ef4444', '#10b981', '#8b5cf6', '#f59e0b', '#06b6d4', 
                        '#ec4899', '#84cc16', '#f97316', '#6366f1', '#14b8a6',
                        '#f43f5e', '#22c55e', '#a855f7', '#eab308', '#0ea5e9']

    unique_names = summary_table['name'].unique()
    name_main_colors = {name: name_colors_base[i % len(name_colors_base)] for i, name in enumerate(unique_names)}
    
    # name별 색상 팔레트 생성 (각 name마다 여러 shade)
    def generate_color_shades(base_color, count):
        """기본 색상에서 여러 shade 생성"""
        import colorsys
        
        # hex to rgb
        base_color = base_color.lstrip('#')
        rgb = tuple(int(base_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        hsv = colorsys.rgb_to_hsv(*rgb)
        
        shades = []
        for i in range(count):
            # 밝기를 조정해서 다른 shade 생성
            new_v = max(0.4, min(0.9, hsv[2] - 0.15 + (i * 0.3 / count)))
            new_s = max(0.5, min(1.0, hsv[1] + (i * 0.2 / count)))
            new_rgb = colorsys.hsv_to_rgb(hsv[0], new_s, new_v)
            hex_color = '#%02x%02x%02x' % tuple(int(c * 255) for c in new_rgb)
            shades.append(hex_color)
        return shades

    # 트리 데이터 구조 생성
    tree_data = {
        'id': 'root',
        'name': '주요 응답',
        'color': '#808080',
        'expanded': False,
        'children': []
    }

    # name별로 최상위 브랜치 노드 생성
    for name_idx, name in enumerate(unique_names):
        if pd.isna(name):
            name_display = '이름 없음'
            name_id = 'no_name'
        else:
            name_display = str(name)
            name_id = f"name_{name_display.replace(' ', '_')}"

        name_data = summary_table[summary_table['name'] == name]
        unique_keywords_in_name = name_data['keywords'].unique()
        
        # 해당 name의 키워드들에 사용할 색상 생성
        keyword_colors = {}
        if len(unique_keywords_in_name) > 1:
            color_shades = generate_color_shades(name_main_colors[name], len(unique_keywords_in_name))
            for kw_idx, keyword in enumerate(unique_keywords_in_name):
                keyword_colors[keyword] = color_shades[kw_idx]
        else:
            keyword_colors[unique_keywords_in_name[0]] = name_main_colors[name]

        # 키워드별로 두 번째 레벨 브랜치 생성
        keyword_children = []
        for keyword in unique_keywords_in_name:
            if pd.isna(keyword):
                keyword_name = '키워드 없음'
                keyword_id = 'no_keyword'
            else:
                keyword_name = str(keyword)
                keyword_id = f"keyword_{keyword_name.replace(' ', '_')}"

            keyword_summaries = name_data[name_data['keywords'] == keyword]

            # 해당 키워드의 summary들을 children으로 생성
            summary_children = []
            for _, row in keyword_summaries.iterrows():
                summary_name = str(row['summary']) if pd.notna(row['summary']) else '요약 없음'
                summary_id = f"summary_{len(summary_children)}"

                summary_children.append({
                    'id': f"{name_id}_{keyword_id}_{summary_id}",
                    'name': summary_name,
                    'color': keyword_colors[keyword],
                    'cnt': int(row['cnt']),
                    'type': 'summary'
                })

            # 키워드 브랜치 노드 생성
            keyword_node = {
                'id': f"{name_id}_{keyword_id}",
                'name': keyword_name,
                'color': keyword_colors[keyword],
                'expanded': False,
                'cnt': keyword_totals[(name, keyword)],
                'children': summary_children,
                'type': 'keyword'
            }

            keyword_children.append(keyword_node)

        # cnt 값을 기준으로 키워드 정렬
        keyword_children.sort(key=lambda x: x['cnt'], reverse=True)

        # name 브랜치 노드 생성 (최상위)
        name_node = {
            'id': name_id,
            'name': name_display,
            'color': name_main_colors[name],
            'expanded': False,
            'cnt': name_totals[name],
            'children': keyword_children,
            'type': 'name'
        }

        tree_data['children'].append(name_node)

    # cnt 값을 기준으로 name 정렬 (큰 값부터)
    tree_data['children'].sort(key=lambda x: x['cnt'], reverse=True)

    return tree_data

# 4단계 하이브리드 마인드맵 생성 함수 (메인 → name → keywords → summary)
def create_tree_force_hybrid_mindmap_4level(tree_data):
    """
    트리 형태 + Force Simulation 하이브리드 마인드맵 (4단계)
    - 기본 형태: 좌측에서 우측으로 펼쳐지는 트리
    - 동적 기능: 겹침 방지, 드래그, 줌 등
    """
    # 최대/최소 cnt 값으로 노드 크기 정규화
    all_cnts = []
    def collect_cnts(node):
        if 'cnt' in node:
            all_cnts.append(node['cnt'])
        if 'children' in node:
            for child in node['children']:
                collect_cnts(child)

    collect_cnts(tree_data)
    max_cnt = max(all_cnts) if all_cnts else 1
    min_cnt = min(all_cnts) if all_cnts else 1

    # HTML/CSS/JavaScript 코드
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>4-Level Tree + Force Hybrid MindMap</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                overflow: hidden;
                height: 100vh;
            }}
            
            .mindmap-container {{
                position: relative;
                width: 100%;
                height: 100vh;
                background: white;
                overflow: hidden;
            }}
            
            .controls {{
                position: absolute;
                top: 20px;
                left: 20px;
                z-index: 1000;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
            }}
            
            .control-btn {{
                background: rgba(59, 130, 246, 0.9);
                color: white;
                border: none;
                padding: 8px 14px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 600;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }}
            
            .control-btn:hover {{
                background: rgba(59, 130, 246, 1);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            }}
            
            .info-panel {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255, 255, 255, 0.95);
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                min-width: 220px;
                backdrop-filter: blur(10px);
                font-size: 13px;
            }}
            
            .info-panel h3 {{
                margin-bottom: 10px;
                color: #1e293b;
                font-size: 16px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 6px 0;
                font-size: 13px;
            }}
            
            .legend-color {{
                width: 14px;
                height: 14px;
                border-radius: 50%;
                margin-right: 8px;
                border: 2px solid white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            
            .svg-container {{
                width: 100%;
                height: 100%;
            }}
            
            .node-group {{
                cursor: pointer;
            }}
            
            .node-circle {{
                stroke: #fff;
                stroke-width: 3px;
                transition: all 0.3s ease;
                filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2));
            }}
            
            .node-circle:hover {{
                stroke-width: 5px;
                filter: drop-shadow(0 6px 12px rgba(0,0,0,0.3)) brightness(1.1);
            }}
            
            .node-rect {{
                stroke: #fff;
                stroke-width: 2px;
                transition: all 0.3s ease;
                filter: drop-shadow(0 3px 8px rgba(0,0,0,0.2));
            }}
            
            .node-rect:hover {{
                stroke-width: 4px;
                filter: drop-shadow(0 5px 12px rgba(0,0,0,0.3)) brightness(1.1);
            }}
            
            .node-text {{
                font-family: 'Segoe UI', sans-serif;
                text-anchor: middle;
                dominant-baseline: middle;
                font-weight: 700;
                fill: white;
                pointer-events: none;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
            }}
            
            .node-count {{
                font-family: 'Segoe UI', sans-serif;
                text-anchor: middle;
                dominant-baseline: middle;
                font-size: 10px;
                fill: rgba(255,255,255,0.9);
                pointer-events: none;
                font-weight: 600;
            }}
            
            .link {{
                fill: none;
                stroke-width: 2;
                opacity: 0.8;
                filter: drop-shadow(0 1px 2px rgba(0,0,0,0.1));
            }}
            
            .link-main {{
                stroke: #64748b;
                stroke-width: 5;
                opacity: 1;
            }}
            
            .link-name {{
                stroke-width: 4;
                opacity: 0.9;
            }}
            
            .link-keyword {{
                stroke-width: 3;
                opacity: 0.8;
            }}
            
            .link-summary {{
                stroke-width: 2;
                opacity: 0.7;
            }}
            
            .tooltip {{
                position: absolute;
                background: rgba(30, 41, 59, 0.95);
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 13px;
                pointer-events: none;
                z-index: 2000;
                max-width: 300px;
                word-wrap: break-word;
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(15px);
                opacity: 0;
                transition: opacity 0.3s ease;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .zoom-indicator {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                background: rgba(255, 255, 255, 0.9);
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 13px;
                color: #64748b;
                z-index: 1000;
                font-weight: 600;
                backdrop-filter: blur(10px);
            }}
            
            .status-indicator {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                background: rgba(16, 185, 129, 0.9);
                color: white;
                padding: 8px 12px;
                border-radius: 8px;
                font-size: 12px;
                z-index: 1000;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="mindmap-container" id="mindmapContainer">
            <!-- 컨트롤 버튼들 -->
            <div class="controls">
                <button class="control-btn" onclick="toggleExpansion()">전체 펼치기/접기</button>
                <button class="control-btn" onclick="resetZoom()">줌 리셋</button>
            </div>
            
            <!-- 정보 패널 -->
            <div class="info-panel">
                <h3>키워드 마인드맵</h3>
                <div style="margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 12px; color: #64748b; line-height: 1.4;">
                    <strong>4단계 구조:</strong><br>
                    메인 → 상위개념 → 키워드 → 요약<br>
                    • 드래그로 자유 이동<br>
                </div>
            </div>
            
            <!-- SVG 컨테이너 -->
            <svg class="svg-container" id="svg"></svg>
            
            <!-- 상태 표시기들 -->
            <div class="zoom-indicator" id="zoomIndicator">줌: 100%</div>
        </div>
        
        <!-- 툴팁 -->
        <div id="tooltip" class="tooltip"></div>

        <script>
            // 전역 변수들
            let svg, g, simulation, nodes, links;
            let allExpanded = false;
            let physicsEnabled = true;
            
            // 트리 데이터
            const treeData = {json.dumps(tree_data, ensure_ascii=False)};
            const maxCnt = {max_cnt};
            const minCnt = {min_cnt};
            
            // 색상 매핑
            const typeColors = {{
                'root': '#3b82f6',
                'name': '#ef4444',
                'keyword': '#10b981',
                'summary': '#8b5cf6'
            }};
            
            // 노드 크기 계산 함수
            function getNodeRadius(cnt, type) {{
                const normalizedCnt = cnt ? (cnt - minCnt) / (maxCnt - minCnt) : 0.3;
                
                switch(type) {{
                    case 'root': return 40;
                    case 'name': return 28 + (normalizedCnt * 18);
                    case 'keyword': return 22 + (normalizedCnt * 14);
                    case 'summary': return 15 + (normalizedCnt * 10);
                    default: return 18;
                }}
            }}
            
            // Summary 노드 크기 계산 함수 (직사각형)
            function getSummaryNodeSize(text, cnt) {{
                const normalizedCnt = cnt ? (cnt - minCnt) / (maxCnt - minCnt) : 0.3;
            
                const maxWidth = 300;  // 한 줄 최대 폭
                const minWidth = 120;
                const padding = 20;
            
                // 글자 수에 따라 width 계산
                const approxWidth = text.length * 7; // 글자 폭 추정
                const width = Math.min(maxWidth, Math.max(minWidth, approxWidth + padding));
            
                // --- 🔑 자동 줄바꿈 시 필요한 line 수 추정 ---
                const charsPerLine = Math.floor((width - padding) / 7);
                const lines = Math.ceil(text.length / charsPerLine);
            
                const lineHeight = 18;
                const height = lines * lineHeight + 20; // 줄 수에 맞게 높이 조정
            
                return {{ width: width, height: height }};
            }}
            
            // 트리 목표 위치 계산 함수
            function calculateTreePositions(nodes) {{
                const width = window.innerWidth;
                const height = window.innerHeight;
                const rootX = 120;
                const rootY = height / 2;
                
                // 깊이별 X 위치 (4단계)
                const depthXPositions = {{
                    0: rootX,           // root
                    1: rootX + 180,     // name
                    2: rootX + 360,     // keywords
                    3: rootX + 600      // summary (더 오른쪽으로)
                }};
                
                // 각 깊이별 노드들 그룹화
                const nodesByDepth = {{}};
                nodes.forEach(node => {{
                    if (!nodesByDepth[node.depth]) {{
                        nodesByDepth[node.depth] = [];
                    }}
                    nodesByDepth[node.depth].push(node);
                }});
                
                // 각 깊이별로 Y 위치 계산
                Object.keys(nodesByDepth).forEach(depth => {{
                    const depthNodes = nodesByDepth[depth];
                    const depthInt = parseInt(depth);
                    
                    if (depthInt === 0) {{
                        // Root 노드는 중앙에
                        depthNodes[0].targetX = rootX;
                        depthNodes[0].targetY = rootY;
                    }} else {{
                        // 부모 노드 기준으로 자식들 배치
                        depthNodes.forEach((node, index) => {{
                            const parent = nodes.find(n => n.id === node.parent);
                            if (parent) {{
                                // 같은 부모를 가진 형제 노드들 찾기
                                const siblings = depthNodes.filter(n => n.parent === node.parent);
                                const siblingIndex = siblings.indexOf(node);
                                const siblingCount = siblings.length;
                                
                                // 형제 노드들의 Y 위치 계산
                                const spacing = Math.max(40, Math.min(80, height / (siblingCount + 2)));
                                const startY = parent.targetY - (siblingCount - 1) * spacing / 2;
                                
                                node.targetX = depthXPositions[depthInt];
                                node.targetY = startY + siblingIndex * spacing;
                            }} else {{
                                // 부모가 없는 경우 기본 위치
                                const spacing = height / (depthNodes.length + 1);
                                node.targetX = depthXPositions[depthInt];
                                node.targetY = spacing * (index + 1);
                            }}
                        }});
                    }}
                }});
            }}
            
            // 트리 데이터를 플랫 구조로 변환
            function flattenTreeData(node, parent = null, depth = 0) {{
                const result = [];
                const currentNode = {{
                    id: node.id,
                    name: node.name,
                    type: node.type || (node.id === 'root' ? 'root' : 'unknown'),
                    cnt: node.cnt || 0,
                    color: node.color || typeColors[node.type] || typeColors['summary'],
                    parent: parent,
                    depth: depth,
                    expanded: node.expanded || false,
                    originalChildren: node.children || [],
                    // 초기 위치 (나중에 계산됨)
                    targetX: 0,
                    targetY: 0
                }};
                
                result.push(currentNode);
                
                // 노드가 확장되어 있으면 자식들도 포함
                if (node.expanded && node.children) {{
                    node.children.forEach(child => {{
                        result.push(...flattenTreeData(child, node.id, depth + 1));
                    }});
                }}
                
                return result;
            }}
            
            // 링크 생성
            function createLinks(nodes) {{
                const links = [];
                nodes.forEach(node => {{
                    if (node.parent) {{
                        const parentNode = nodes.find(n => n.id === node.parent);
                        if (parentNode) {{
                            links.push({{
                                source: node.parent,
                                target: node.id,
                                type: `link-${{node.type}}`
                            }});
                        }}
                    }}
                }});
                return links;
            }}
            
            // 초기화 함수
            function initializeMindMap() {{
                const container = d3.select("#mindmapContainer");
                const width = container.node().clientWidth;
                const height = container.node().clientHeight;
                
                // SVG 설정
                svg = d3.select("#svg")
                    .attr("width", width)
                    .attr("height", height);
                
                // 줌 기능 설정
                const zoom = d3.zoom()
                    .scaleExtent([0.1, 4])
                    .on("zoom", (event) => {{
                        g.attr("transform", event.transform);
                        d3.select("#zoomIndicator").text(`줌: ${{Math.round(event.transform.k * 100)}}%`);
                    }});
                
                svg.call(zoom);
                
                // 메인 그룹
                g = svg.append("g");
                
                // 루트 노드만 확장된 상태로 시작
                treeData.expanded = true;
                
                // 초기 데이터 로드
                updateVisualization();
            }}
            
            // 시각화 업데이트
            function updateVisualization() {{
                // 노드와 링크 데이터 생성
                nodes = flattenTreeData(treeData);
                links = createLinks(nodes);
                
                // 트리 목표 위치 계산
                calculateTreePositions(nodes);
                
                // 초기 위치를 목표 위치로 설정 (새 노드들만)
                nodes.forEach(node => {{
                    if (node.x === undefined) {{
                        node.x = node.targetX;
                        node.y = node.targetY;
                    }}
                }});
                
                // 시뮬레이션 설정
                simulation = d3.forceSimulation(nodes)
                    .force("link", d3.forceLink(links).id(d => d.id).distance(d => {{
                        const targetNode = nodes.find(n => n.id === d.target.id || n.id === d.target);
                        return targetNode ? 60 + (targetNode.depth * 25) : 80;
                    }}))
                    .force("charge", d3.forceManyBody()
                        .strength(d => {{
                            if (!physicsEnabled) return 0;
                            // 노드 타입에 따른 약한 반발력 (트리 구조 유지용)
                            switch(d.type) {{
                                case 'root': return -350;
                                case 'name': return -250;
                                case 'keyword': return -180;
                                case 'summary': return -100;
                                default: return -150;
                            }}
                        }})
                    )
                    // 목표 위치로 끌어당기는 힘 (트리 구조 유지)
                    .force("x", d3.forceX(d => d.targetX).strength(0.3))
                    .force("y", d3.forceY(d => d.targetY).strength(0.3))
                    // 충돌 방지 (겹침 방지) - summary 노드는 직사각형 고려
                    .force("collision", d3.forceCollide().radius(d => {{
                        if (d.type === 'summary') {{
                            const summarySize = getSummaryNodeSize(d.name, d.cnt);
                            return Math.max(summarySize.width, summarySize.height) / 2 + 8;
                        }}
                        return getNodeRadius(d.cnt, d.type) + 5;
                    }}));
                
                // 링크 렌더링
                const link = g.selectAll(".link")
                    .data(links, d => `${{d.source.id || d.source}}-${{d.target.id || d.target}}`);
                
                link.exit().transition().duration(300).style("opacity", 0).remove();
                
                const linkEnter = link.enter().append("line")
                    .attr("class", d => `link ${{d.type}}`)
                    .style("opacity", 0);
                
                const linkUpdate = linkEnter.merge(link)
                    .transition().duration(300).style("opacity", 0.8)
                    .attr("stroke", d => {{
                        const targetNode = nodes.find(n => n.id === d.target.id || n.id === d.target);
                        return targetNode ? targetNode.color : '#64748b';
                    }});
                
                // 노드 그룹 렌더링
                const node = g.selectAll(".node-group")
                    .data(nodes, d => d.id);
                
                node.exit().transition().duration(300).style("opacity", 0).remove();
                
                const nodeEnter = node.enter().append("g")
                    .attr("class", "node-group")
                    .style("opacity", 0)
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended)
                    );
                
                // 노드 원 추가 (summary는 직사각형)
                nodeEnter.each(function(d) {{
                    const nodeGroup = d3.select(this);
                    if (d.type === 'summary') {{
                        // ✅ summary 전용 foreignObject 사용
                        const size = getSummaryNodeSize(d.name, d.cnt);
                
                        nodeGroup.append("rect")
                            .attr("class", "node-rect summary-rect")
                            .attr("width", size.width)
                            .attr("height", size.height)
                            .attr("x", -size.width / 2)
                            .attr("y", -size.height / 2)
                            .attr("rx", 8)
                            .attr("ry", 8)
                            .attr("fill", d.color);
                
                        nodeGroup.append("foreignObject")
                            .attr("x", -size.width / 2 + 6)
                            .attr("y", -size.height / 2 + 6)
                            .attr("width", size.width - 12)
                            .attr("height", size.height - 12)
                            .append("xhtml:div")
                            .style("width", (size.width - 12) + "px")
                            .style("height", (size.height - 12) + "px")
                            .style("font-size", "12px")
                            .style("line-height", "1.4em")
                            .style("color", "white")
                            .style("font-family", "Segoe UI, sans-serif")
                            .style("text-align", "center")
                            .style("word-wrap", "break-word")
                            .style("overflow-wrap", "break-word")
                            .style("display", "flex")
                            .style("align-items", "center")
                            .style("justify-content", "center")
                            .text(d.name);
                
                    }} else {{
                        // ✅ summary가 아닌 경우만 circle + node-text 사용
                        nodeGroup.append("circle").attr("class", "node-circle");
                
                        nodeGroup.append("text")
                            .attr("class", "node-text")
                            .text(d => d.name);
                    }}
                }});
                
                // 노드 텍스트 추가
                nodeEnter.append("text")
                    .attr("class", "node-text");
                
                // 노드 카운트 텍스트 추가  
                nodeEnter.append("text")
                    .attr("class", "node-count")
                    .attr("dy", "1.5em");
                
                const nodeUpdate = nodeEnter.merge(node)
                    .transition().duration(300).style("opacity", 1);
                
                // 노드 원/직사각형 업데이트
                nodeUpdate.each(function(d) {{
                    const nodeGroup = d3.select(this);
                    if (d.type === 'summary') {{
                        // Summary 노드 직사각형 업데이트
                        const summarySize = getSummaryNodeSize(d.name, d.cnt);
                        nodeGroup.select(".node-rect")
                            .transition().duration(300)
                            .attr("width", summarySize.width)
                            .attr("height", summarySize.height)
                            .attr("x", -summarySize.width / 2)
                            .attr("y", -summarySize.height / 2)
                            .attr("rx", 8)
                            .attr("ry", 8)
                            .attr("fill", d.color)
                            .attr("stroke", "#fff")
                            .attr("stroke-width", 2);
                    }} else {{
                        // 원형 노드 업데이트
                        nodeGroup.select(".node-circle")
                            .transition().duration(300)
                            .attr("r", getNodeRadius(d.cnt, d.type))
                            .attr("fill", d.color)
                            .attr("stroke-width", d => {{
                                switch(d.type) {{
                                    case 'root': return 5;
                                    case 'name': return 4;
                                    case 'keyword': return 3;
                                    default: return 3;
                                }}
                            }});
                    }}
                }});
                
                // 노드 텍스트 업데이트
                nodeUpdate.select(".node-text")
                    .filter(d => d.type !== "summary")   // ✅ summary 제외
                    .text(d => {{
                        const maxLength = Math.max(5, Math.floor(getNodeRadius(d.cnt, d.type) / 3));
                        return d.name.length > maxLength ? 
                               d.name.substring(0, maxLength) + '...' : 
                               d.name;
                    }})
                    .attr("font-size", d => Math.max(9, getNodeRadius(d.cnt, d.type) / 2.2) + "px");
                
                // 노드 카운트 업데이트
                nodeUpdate.select(".node-count")
                    .filter(d => d.type !== "summary")   // ✅ summary 제외
                    .text(d => d.cnt ? `${{d.cnt}}명` : '')
                    .attr("font-size", "10px");
                
                // 이벤트 핸들러
                g.selectAll(".node-group")
                    .on("click", function(event, d) {{
                        if (d.originalChildren && d.originalChildren.length > 0) {{
                            toggleNode(d);
                        }}
                    }})
                    .on("mouseover", function(event, d) {{
                        showTooltip(event, d);
                        if (d.type === 'summary') {{
                            // Summary 노드 (직사각형) 호버 효과
                            d3.select(this).select(".node-rect").transition().duration(200)
                                .attr("stroke-width", 4);
                        }} else {{
                            // 원형 노드 호버 효과
                            d3.select(this).select(".node-circle").transition().duration(200)
                                .attr("stroke-width", d => {{
                                    switch(d.type) {{
                                        case 'root': return 7;
                                        case 'name': return 6;
                                        case 'keyword': return 5;
                                        default: return 5;
                                    }}
                                }});
                        }}
                    }})
                    .on("mouseout", function(event, d) {{
                        hideTooltip();
                        if (d.type === 'summary') {{
                            // Summary 노드 (직사각형) 호버 해제
                            d3.select(this).select(".node-rect").transition().duration(200)
                                .attr("stroke-width", 2);
                        }} else {{
                            // 원형 노드 호버 해제
                            d3.select(this).select(".node-circle").transition().duration(200)
                                .attr("stroke-width", d => {{
                                    switch(d.type) {{
                                        case 'root': return 5;
                                        case 'name': return 4;
                                        case 'keyword': return 3;
                                        default: return 3;
                                    }}
                                }});
                        }}
                    }});
                
                // 시뮬레이션 틱 이벤트
                simulation.on("tick", () => {{
                    g.selectAll(".link")
                        .attr("x1", d => d.source.x)
                        .attr("y1", d => d.source.y)
                        .attr("x2", d => d.target.x)
                        .attr("y2", d => d.target.y);
                    
                    g.selectAll(".node-group")
                        .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
                }});
                
                // 시뮬레이션 재시작
                simulation.alpha(1).restart();
            }}
            
            // 노드 토글
            function toggleNode(node) {{
                const originalNode = findNodeInTree(treeData, node.id);
                if (originalNode) {{
                    originalNode.expanded = !originalNode.expanded;
                    updateVisualization();
                }}
            }}
            
            // 트리에서 노드 찾기
            function findNodeInTree(tree, nodeId) {{
                if (tree.id === nodeId) return tree;
                if (tree.children) {{
                    for (let child of tree.children) {{
                        const found = findNodeInTree(child, nodeId);
                        if (found) return found;
                    }}
                }}
                return null;
            }}
            
            // 드래그 이벤트 핸들러들
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}
            
            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}
            
            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}
            
            // 툴팁 기능
            function showTooltip(event, node) {{
                const tooltip = d3.select("#tooltip");
                
                let content = `<strong>${{node.name}}</strong><br>`;
                content += `응답자 수: ${{node.cnt}}명<br>`;
                content += `노드 타입: ${{node.type}}<br>`;
                content += `계층 깊이: ${{node.depth}}단계`;
                
                if (node.originalChildren && node.originalChildren.length > 0) {{
                    content += `<br>하위 노드: ${{node.originalChildren.length}}개`;
                    content += `<br>클릭해서 ${{node.expanded ? '접기' : '펼치기'}}`;
                }}
                
                tooltip.html(content)
                    .style("left", (event.pageX + 20) + "px")
                    .style("top", (event.pageY + 10) + "px")
                    .style("opacity", 1);
            }}
            
            function hideTooltip() {{
                d3.select("#tooltip").style("opacity", 0);
            }}
            
            // 컨트롤 함수들
            function toggleExpansion() {{
                allExpanded = !allExpanded;
                expandAll(treeData, allExpanded);
                updateVisualization();
            }}
            
            function expandAll(node, expand) {{
                node.expanded = expand;
                if (node.children) {{
                    node.children.forEach(child => expandAll(child, expand));
                }}
            }}
            
            function resetZoom() {{
                svg.transition().duration(750).call(
                    d3.zoom().transform,
                    d3.zoomIdentity
                );
            }}
            
            function resetTreeLayout() {{
                // 모든 노드의 고정 위치 해제
                nodes.forEach(node => {{
                    node.fx = null;
                    node.fy = null;
                }});
                
                // 목표 위치 재계산
                calculateTreePositions(nodes);
                
                // 시뮬레이션에서 목표 위치로 이동
                simulation.force("x", d3.forceX(d => d.targetX).strength(0.5));
                simulation.force("y", d3.forceY(d => d.targetY).strength(0.5));
                simulation.alpha(1).restart();
                
                // 1초 후에 원래 강도로 복원
                setTimeout(() => {{
                    simulation.force("x", d3.forceX(d => d.targetX).strength(0.3));
                    simulation.force("y", d3.forceY(d => d.targetY).strength(0.3));
                }}, 1000);
            }}
            
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                
                const statusIndicator = d3.select("#statusIndicator");
                if (physicsEnabled) {{
                    statusIndicator.text("4단계 트리 + 물리엔진").style("background", "rgba(16, 185, 129, 0.9)");
                    simulation.force("charge", d3.forceManyBody().strength(d => {{
                        switch(d.type) {{
                            case 'root': return -350;
                            case 'name': return -250;
                            case 'keyword': return -180;
                            case 'summary': return -100;
                            default: return -150;
                        }}
                    }}));
                }} else {{
                    statusIndicator.text("4단계 트리 전용").style("background", "rgba(239, 68, 68, 0.9)");
                    simulation.force("charge", d3.forceManyBody().strength(0));
                }}
                
                simulation.alpha(0.3).restart();
            }}
            
            // 창 크기 변경 대응
            window.addEventListener('resize', () => {{
                const container = d3.select("#mindmapContainer");
                const width = container.node().clientWidth;
                const height = container.node().clientHeight;
                
                svg.attr("width", width).attr("height", height);
                
                if (simulation && nodes) {{
                    // 목표 위치 재계산
                    calculateTreePositions(nodes);
                    simulation.force("x", d3.forceX(d => d.targetX).strength(0.3));
                    simulation.force("y", d3.forceY(d => d.targetY).strength(0.3));
                    simulation.alpha(0.3).restart();
                }}
            }});
            
            // 초기화
            document.addEventListener('DOMContentLoaded', initializeMindMap);
        </script>
    </body>
    </html>
    """

    return html_code

def main():
    st.set_page_config(
        page_title="MindMap & RAG Chatbot",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 키워드 마인드맵 + RAG 질의응답")
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("전체 응답 수", df[df.total_cl != 99].user_id.nunique())
                with col2:
                    st.metric("전체 청크 수", len(df))
                with col3:
                    st.metric("키워드 분류 청크 수", len(filtered_df))

            
            # 왼쪽/오른쪽 분할 레이아웃
            left_col, right_col = st.columns([1, 1])

            # 마인드맵 생성
            if has_mindmap_columns:
                tree_data = create_tree_data_from_csv_4level(df)

                with left_col:
                    st.subheader("🗺️ 키워드 마인드맵")
                    st.markdown("*메인 → 상위주제 → 키워드 → 요약*")

                    # 4단계 하이브리드 마인드맵 시각화
                    html_code = create_tree_force_hybrid_mindmap_4level(tree_data)
                    components.html(html_code, height=600, scrolling=False)

                    filtered_df = df[df.total_cl != 99]
                    
                    st.subheader("📋 키워드 별 관련 청크")
                    st.text("키워드로 분류된 청크들을 확인할 수 있습니다.(테이블 우측 상단 다운로드 가능)")
                    summary_table = (
                        filtered_df
                        .groupby(['name', 'keywords', 'summary'], as_index=False, dropna=False)
                        .agg(
                            cnt=('user_id', 'nunique'),
                            answers=('SPLITTED', lambda x: list(x.dropna()))
                        )
                    )
                    st.dataframe(
                        summary_table.sort_values(['name','keywords'], ascending=False), 
                        use_container_width=True,
                        #height=500
                    )
            
            else:
                with left_col:
                    st.info(" 마인드맵 생성을 위해서는 user_id, total_cl, name, keywords, summary 컬럼이 필요합니다.")

            with right_col:

                if has_mindmap_columns:
                    # Summary Table (4단계 구조)
                    st.subheader("📋 키워드 미분류 청크")
                    st.text("키워드로 분류되지 않은 청크들을 확인할 수 있습니다.(테이블 우측 상단 다운로드 가능)")
                    no_filtered_df = df[df.total_cl == 99][["user_id","SPLITTED"]]
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
                                        score = doc.metadata.get('score', None)
                                        source_filename = os.path.basename(source)
                                    
                                        st.markdown(f"👤 {source_filename} 📊 유사도: {score:.2f}")
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
