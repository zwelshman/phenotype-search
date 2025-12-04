import streamlit as st
import pandas as pd
from pyconceptlibraryclient import Client
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import io
from datetime import datetime

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'selected_phenotypes' not in st.session_state:
    st.session_state.selected_phenotypes = set()
if 'embedding_model' not in st.session_state:
    with st.spinner('Loading embedding model (first time only)...'):
        st.session_state.embedding_model = SentenceTransformer('BAAI/llm-embedder')

# Page config
st.set_page_config(
    page_title="HDR UK Phenotype Search",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 HDR UK Phenotype Library Search")
st.markdown("Search for phenotypes using natural language queries and download or combine codelists")

# Initialize HDR UK client
@st.cache_resource
def get_hdr_client():
    """Get authenticated HDR UK client."""
    return Client(public=True, url="https://phenotypes.healthdatagateway.org/")

client = get_hdr_client()

def create_searchable_text(phenotype: Dict) -> str:
    """Create rich text for semantic search."""
    parts = [
        phenotype.get('name', ''),
        phenotype.get('description', ''),
        phenotype.get('phenotype_type', ''),
        ' '.join(phenotype.get('author', [])) if phenotype.get('author') else '',
        phenotype.get('coding_system', ''),
    ]
    return ' '.join(filter(None, parts))

def semantic_rerank(query: str, results: List[Dict], top_k: int = 20) -> List[Dict]:
    """Rerank results using semantic similarity."""
    if not results:
        return []
    
    # Create searchable text for each result
    result_texts = [create_searchable_text(r) for r in results]
    
    # Encode query and results
    query_embedding = st.session_state.embedding_model.encode([query])[0]
    result_embeddings = st.session_state.embedding_model.encode(result_texts)
    
    # Calculate cosine similarity
    similarities = np.dot(result_embeddings, query_embedding) / (
        np.linalg.norm(result_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Add similarity scores and sort
    for i, result in enumerate(results):
        result['similarity_score'] = float(similarities[i])
    
    results_sorted = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    return results_sorted[:top_k]

def search_phenotypes(query: str, max_results: int = 50) -> List[Dict]:
    """Search phenotypes with semantic reranking."""
    try:
        # First get results from HDR UK API
        api_results = list(client.phenotypes.get(search=query))

        if not api_results:
            return []

        # Limit to max_results for semantic reranking
        api_results = api_results[:max_results]
        
        # Rerank using semantic similarity
        reranked = semantic_rerank(query, api_results, top_k=20)
        
        return reranked
    except Exception as e:
        st.error(f"Error searching phenotypes: {str(e)}")
        return []

def get_phenotype_detail(phenotype_id: str) -> Dict:
    """Get full phenotype details."""
    try:
        return client.phenotypes.get_detail(phenotype_id)
    except Exception as e:
        st.error(f"Error fetching phenotype details: {str(e)}")
        return {}

def get_phenotype_codelist(phenotype_id: str) -> pd.DataFrame:
    """Get codelist for a phenotype."""
    try:
        codes = client.phenotypes.get_codelist(phenotype_id)
        if codes:
            return pd.DataFrame(codes)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching codelist: {str(e)}")
        return pd.DataFrame()

def download_phenotype(phenotype_id: str, phenotype_name: str):
    """Create downloadable CSV for a phenotype."""
    codelist = get_phenotype_codelist(phenotype_id)
    
    if not codelist.empty:
        # Create CSV in memory
        csv_buffer = io.StringIO()
        codelist.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()
        
        # Create download button
        st.download_button(
            label=f"📥 Download {phenotype_name}",
            data=csv_str,
            file_name=f"{phenotype_id}_{phenotype_name.replace(' ', '_')}.csv",
            mime="text/csv",
            key=f"download_{phenotype_id}"
        )
    else:
        st.warning("No codes available for this phenotype")

def combine_phenotypes(phenotype_ids: List[str], phenotype_names: List[str]) -> pd.DataFrame:
    """Combine codelists from multiple phenotypes."""
    combined_codes = []
    
    progress_bar = st.progress(0)
    for i, (pid, pname) in enumerate(zip(phenotype_ids, phenotype_names)):
        codelist = get_phenotype_codelist(pid)
        if not codelist.empty:
            codelist['source_phenotype_id'] = pid
            codelist['source_phenotype_name'] = pname
            combined_codes.append(codelist)
        progress_bar.progress((i + 1) / len(phenotype_ids))
    
    if combined_codes:
        combined_df = pd.concat(combined_codes, ignore_index=True)
        # Remove duplicate codes
        if 'code' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['code'], keep='first')
        return combined_df
    return pd.DataFrame()

# Search interface
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "🔍 Search for phenotypes",
        placeholder="e.g., type 2 diabetes, myocardial infarction, chronic kidney disease",
        help="Enter a natural language description of the phenotype you're looking for"
    )

with col2:
    max_results = st.selectbox("Max results", [10, 20, 30, 50], index=1)

search_button = st.button("Search", type="primary", use_container_width=True)

# Perform search
if search_button and query:
    with st.spinner('Searching phenotypes...'):
        st.session_state.search_results = search_phenotypes(query, max_results=max_results)
        st.session_state.selected_phenotypes = set()

# Display results
if st.session_state.search_results:
    st.markdown("---")
    st.subheader(f"Found {len(st.session_state.search_results)} phenotypes")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Clear Selection", use_container_width=True):
            st.session_state.selected_phenotypes = set()
            st.rerun()
    with col2:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_phenotypes = {r['phenotype_id'] for r in st.session_state.search_results}
            st.rerun()
    
    # Display each result
    for result in st.session_state.search_results:
        phenotype_id = result['phenotype_id']
        
        with st.expander(
            f"**{result['name']}** ({phenotype_id}) - Similarity: {result.get('similarity_score', 0):.3f}",
            expanded=False
        ):
            # Checkbox for selection
            is_selected = st.checkbox(
                "Select this phenotype",
                key=f"select_{phenotype_id}",
                value=phenotype_id in st.session_state.selected_phenotypes
            )
            
            if is_selected:
                st.session_state.selected_phenotypes.add(phenotype_id)
            else:
                st.session_state.selected_phenotypes.discard(phenotype_id)
            
            # Display metadata
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Type:** {result.get('phenotype_type', 'N/A')}")
                st.markdown(f"**Coding System:** {result.get('coding_system', 'N/A')}")
                if result.get('author'):
                    st.markdown(f"**Authors:** {', '.join(result['author'][:3])}")
            with col2:
                if result.get('collections'):
                    collection_names = []
                    for c in result['collections'][:2]:
                        if isinstance(c, dict):
                            collection_names.append(c.get('name', ''))
                        elif isinstance(c, str):
                            collection_names.append(c)
                    if collection_names:
                        st.markdown(f"**Collections:** {', '.join(collection_names)}")
                if result.get('data_sources'):
                    st.markdown(f"**Data Sources:** {', '.join(result['data_sources'][:3])}")
            
            # Description
            if result.get('description'):
                st.markdown(f"**Description:** {result['description'][:300]}...")
            
            # Action buttons
            col1, col2 = st.columns([1, 3])
            with col1:
                download_phenotype(phenotype_id, result['name'])
            with col2:
                if st.button(f"View Full Details", key=f"details_{phenotype_id}"):
                    detail = get_phenotype_detail(phenotype_id)
                    st.json(detail)
    
    # Combine selected phenotypes
    st.markdown("---")
    st.subheader("Combine Selected Phenotypes")
    
    if st.session_state.selected_phenotypes:
        st.info(f"✅ {len(st.session_state.selected_phenotypes)} phenotype(s) selected")
        
        if st.button("🔗 Combine Selected Phenotypes", type="primary", use_container_width=True):
            with st.spinner('Combining phenotypes...'):
                selected_ids = list(st.session_state.selected_phenotypes)
                selected_names = [
                    r['name'] for r in st.session_state.search_results 
                    if r['phenotype_id'] in selected_ids
                ]
                
                combined_df = combine_phenotypes(selected_ids, selected_names)
                
                if not combined_df.empty:
                    st.success(f"✅ Combined {len(combined_df)} unique codes from {len(selected_ids)} phenotypes")
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Codes", len(combined_df))
                    with col2:
                        st.metric("Source Phenotypes", len(selected_ids))
                    with col3:
                        if 'coding_system' in combined_df.columns:
                            st.metric("Coding Systems", combined_df['coding_system'].nunique())
                    
                    # Display preview
                    st.dataframe(combined_df.head(20), use_container_width=True)
                    
                    # Download combined
                    csv_buffer = io.StringIO()
                    combined_df.to_csv(csv_buffer, index=False)
                    csv_str = csv_buffer.getvalue()
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="📥 Download Combined Codelist",
                        data=csv_str,
                        file_name=f"combined_phenotypes_{timestamp}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                else:
                    st.error("No codes found in selected phenotypes")
    else:
        st.info("Select phenotypes above to combine their codelists")

# Sidebar with info
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This tool searches the HDR UK Phenotype Library using natural language queries.
    
    **Features:**
    - 🔍 Semantic search using BAAI/llm-embedder
    - 📥 Download individual phenotype codelists
    - 🔗 Combine multiple phenotypes into one codelist
    - 📊 View metadata and statistics
    
    **Data Source:**
    [HDR UK Phenotype Library](https://phenotypes.healthdatagateway.org/)
    """)
    
    st.markdown("---")
    st.markdown("### Usage Tips")
    st.markdown("""
    - Use clinical terms: "heart failure", "diabetes"
    - Be specific: "type 2 diabetes" vs "diabetes"
    - Check similarity scores for relevance
    - Combine related phenotypes to create comprehensive codelists
    """)
