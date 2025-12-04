import streamlit as st
import pandas as pd
from pyconceptlibraryclient import Client
from sentence_transformers import SentenceTransformer, util
import numpy as np
from typing import List, Dict, Optional
import io
from datetime import datetime
import traceback

# Page config
st.set_page_config(
    page_title="HDR UK Phenotype Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .phenotype-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    .similarity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        background-color: #28a745;
        color: white;
        font-weight: bold;
        font-size: 0.875rem;
    }
    .stExpander {
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'selected_phenotypes' not in st.session_state:
        st.session_state.selected_phenotypes = {}
    if 'all_phenotypes' not in st.session_state:
        st.session_state.all_phenotypes = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    if 'expanded_phenotypes' not in st.session_state:
        st.session_state.expanded_phenotypes = set()

init_session_state()

# Load embedding model (lighter and faster)
@st.cache_resource
def load_embedding_model():
    """Load a lightweight embedding model for semantic search."""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Failed to load embedding model: {str(e)}")
        return None

# Initialize HDR UK client
@st.cache_resource
def get_hdr_client():
    """Get authenticated HDR UK Phenotype Library client."""
    try:
        return Client(public=True, url="https://phenotypes.healthdatagateway.org/")
    except Exception as e:
        st.error(f"Failed to connect to HDR UK Phenotype Library: {str(e)}")
        return None

def create_searchable_text(phenotype: Dict) -> str:
    """Create rich text for semantic search from phenotype metadata."""
    # Type check: ensure phenotype is a dictionary
    if not isinstance(phenotype, dict):
        return ""

    parts = []

    # Core fields
    if phenotype.get('name'):
        parts.append(phenotype['name'])
    if phenotype.get('description'):
        parts.append(phenotype['description'])

    # Metadata
    if phenotype.get('phenotype_type'):
        parts.append(f"Type: {phenotype['phenotype_type']}")
    if phenotype.get('coding_system'):
        parts.append(f"Coding: {phenotype['coding_system']}")

    # Authors
    if phenotype.get('author'):
        authors = phenotype['author']
        if isinstance(authors, list):
            parts.append(' '.join(authors))
        else:
            parts.append(str(authors))

    # Collections
    if phenotype.get('collections'):
        collections = phenotype['collections']
        if isinstance(collections, list):
            for c in collections:
                if isinstance(c, dict) and c.get('name'):
                    parts.append(c['name'])
                elif isinstance(c, str):
                    parts.append(c)

    # Tags
    if phenotype.get('tags'):
        tags = phenotype['tags']
        if isinstance(tags, list):
            parts.extend([str(t) for t in tags])

    return ' '.join(filter(None, parts))

def semantic_search(query: str, phenotypes: List[Dict], model, top_k: int = 30) -> List[Dict]:
    """Perform semantic search using sentence embeddings."""
    if not phenotypes or not model:
        return []

    try:
        # Filter to ensure all items are dictionaries
        valid_phenotypes = [p for p in phenotypes if isinstance(p, dict)]

        if not valid_phenotypes:
            st.warning("No valid phenotype dictionaries found for semantic search")
            return []

        # Create searchable texts
        phenotype_texts = [create_searchable_text(p) for p in valid_phenotypes]

        # Encode
        query_embedding = model.encode(query, convert_to_tensor=True)
        phenotype_embeddings = model.encode(phenotype_texts, convert_to_tensor=True)

        # Calculate cosine similarity
        similarities = util.cos_sim(query_embedding, phenotype_embeddings)[0]

        # Add scores and sort
        results = []
        for idx, phenotype in enumerate(valid_phenotypes):
            phenotype_copy = phenotype.copy()
            phenotype_copy['similarity_score'] = float(similarities[idx])
            results.append(phenotype_copy)

        # Sort by similarity and return top k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

    except Exception as e:
        st.error(f"Error in semantic search: {str(e)}")
        return valid_phenotypes[:top_k] if valid_phenotypes else []

def search_phenotypes(query: str, client, model, filters: Dict = None) -> List[Dict]:
    """Search phenotypes with optional filters."""
    try:
        # Build search parameters
        search_params = {'search': query, 'no_pagination': True}

        if filters:
            if filters.get('phenotype_type'):
                search_params['phenotype_type'] = filters['phenotype_type']
            if filters.get('collection_id'):
                search_params['collections'] = filters['collection_id']

        # Get results from API
        # With no_pagination=True, the API returns a list of phenotype dictionaries directly
        results = client.phenotypes.get(**search_params)

        # Validate results
        if not results:
            return []

        if not isinstance(results, list):
            st.error(f"Unexpected API response type: {type(results)}. Expected list.")
            return []

        # Filter to ensure all items are dictionaries
        valid_results = [r for r in results if isinstance(r, dict)]

        if not valid_results:
            st.warning("No valid phenotype objects found in API response")
            return []

        # Apply semantic reranking
        reranked = semantic_search(query, valid_results, model, top_k=30)

        return reranked

    except Exception as e:
        st.error(f"Error searching phenotypes: {str(e)}")
        st.error(f"Details: {traceback.format_exc()}")
        return []

def get_phenotype_codelist(client, phenotype_id: str) -> Optional[pd.DataFrame]:
    """Fetch codelist for a phenotype."""
    try:
        codes = client.phenotypes.get_codelist(phenotype_id)
        if codes:
            df = pd.DataFrame(codes)
            return df
        return None
    except Exception as e:
        st.error(f"Error fetching codelist for {phenotype_id}: {str(e)}")
        return None

def download_button(df: pd.DataFrame, filename: str, label: str, key: str):
    """Create a download button for a dataframe."""
    if df is not None and not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label=label,
            data=csv,
            file_name=filename,
            mime='text/csv',
            key=key
        )
    else:
        st.warning("No data available for download")

def display_phenotype_card(phenotype: Dict, idx: int):
    """Display a single phenotype result card."""
    # Type check: ensure phenotype is a dictionary
    if not isinstance(phenotype, dict):
        st.error(f"Invalid phenotype data type: {type(phenotype)}. Expected dictionary.")
        return

    phenotype_id = phenotype.get('phenotype_id', 'Unknown')
    name = phenotype.get('name', 'Unnamed Phenotype')
    similarity = phenotype.get('similarity_score', 0.0)

    # Color code similarity score
    if similarity >= 0.7:
        score_color = "🟢"
    elif similarity >= 0.5:
        score_color = "🟡"
    else:
        score_color = "🔴"

    # Create unique key for this phenotype's expander state
    expander_key = f"{phenotype_id}_{idx}"

    # Check if this phenotype should be expanded
    is_expanded = expander_key in st.session_state.expanded_phenotypes

    with st.expander(f"{score_color} **{name}** (ID: {phenotype_id}) - Match: {similarity:.1%}", expanded=is_expanded):
        # Selection checkbox
        is_selected = st.checkbox(
            "Select for combination",
            key=f"select_{phenotype_id}_{idx}",
            value=phenotype_id in st.session_state.selected_phenotypes
        )

        if is_selected:
            st.session_state.selected_phenotypes[phenotype_id] = name
        else:
            st.session_state.selected_phenotypes.pop(phenotype_id, None)

        # Display metadata in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📋 Metadata**")
            if phenotype.get('phenotype_type'):
                st.markdown(f"- **Type:** {phenotype['phenotype_type']}")
            if phenotype.get('coding_system'):
                st.markdown(f"- **Coding System:** {phenotype['coding_system']}")
            if phenotype.get('sex'):
                st.markdown(f"- **Sex:** {phenotype['sex']}")
            if phenotype.get('created'):
                st.markdown(f"- **Created:** {phenotype['created'][:10]}")

        with col2:
            st.markdown("**👥 Authors & Collections**")
            if phenotype.get('author'):
                authors = phenotype['author']
                if isinstance(authors, list):
                    author_str = ', '.join(authors[:3])
                    if len(authors) > 3:
                        author_str += f" (+{len(authors)-3} more)"
                    st.markdown(f"- **Authors:** {author_str}")

            if phenotype.get('collections'):
                collections = phenotype['collections']
                if isinstance(collections, list) and len(collections) > 0:
                    coll_names = []
                    for c in collections[:2]:
                        if isinstance(c, dict):
                            coll_names.append(c.get('name', ''))
                        elif isinstance(c, str):
                            coll_names.append(c)
                    if coll_names:
                        st.markdown(f"- **Collections:** {', '.join(filter(None, coll_names))}")

        # Description
        if phenotype.get('description'):
            st.markdown("**📝 Description**")
            desc = phenotype['description']
            if len(desc) > 300:
                desc = desc[:300] + "..."
            st.markdown(desc)

        # Tags
        if phenotype.get('tags'):
            tags = phenotype['tags']
            if isinstance(tags, list) and len(tags) > 0:
                tag_str = ', '.join([f"`{t}`" for t in tags[:5]])
                st.markdown(f"**🏷️ Tags:** {tag_str}")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📥 Download Codelist", key=f"dl_{phenotype_id}_{idx}"):
                # Keep expander open
                st.session_state.expanded_phenotypes.add(expander_key)
                with st.spinner(f"Fetching codelist..."):
                    client = get_hdr_client()
                    if client:
                        codelist = get_phenotype_codelist(client, phenotype_id)
                        if codelist is not None and not codelist.empty:
                            st.success(f"Found {len(codelist)} codes")
                            st.dataframe(codelist.head(10), use_container_width=True)
                            download_button(
                                codelist,
                                f"{phenotype_id}_codelist.csv",
                                "💾 Save CSV",
                                f"save_{phenotype_id}_{idx}"
                            )
                        else:
                            st.warning("No codes found")

        with col2:
            if phenotype.get('url'):
                st.link_button("🔗 View on Portal", phenotype['url'], key=f"link_{phenotype_id}_{idx}")

        with col3:
            if st.button("ℹ️ Full Details", key=f"details_{phenotype_id}_{idx}"):
                # Keep expander open
                st.session_state.expanded_phenotypes.add(expander_key)
                with st.spinner("Loading details..."):
                    client = get_hdr_client()
                    if client:
                        try:
                            detail = client.phenotypes.get_detail(phenotype_id)
                            # Display full details in an expandable container for better layout
                            with st.container():
                                st.markdown("**Full Phenotype Details:**")
                                st.json(detail)
                        except Exception as e:
                            st.error(f"Could not fetch details: {str(e)}")

# Main App
st.title("🔬 HDR UK Phenotype Library - Natural Language Search")
st.markdown("""
Search for clinical phenotypes using natural language queries.
Find, compare, and download codelists from the [HDR UK Phenotype Library](https://phenotypes.healthdatagateway.org/).
""")

# Load resources
with st.spinner("🔄 Initializing..."):
    client = get_hdr_client()
    model = load_embedding_model()

if not client or not model:
    st.error("⚠️ Failed to initialize. Please refresh the page or check your connection.")
    st.stop()

# Sidebar for filters and info
with st.sidebar:
    st.header("🔍 Search Filters")

    phenotype_type_filter = st.selectbox(
        "Phenotype Type",
        ["All", "Disease or Syndrome", "Biomarker", "Lifestyle Risk Factor", "Physical Measurement"],
        index=0
    )

    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    **Natural Language Search**
    Uses semantic AI to understand your queries and find the most relevant phenotypes.

    **Features:**
    - 🤖 AI-powered semantic search
    - 🎯 Match scoring for relevance
    - 📊 View and download codelists
    - 🔗 Combine multiple phenotypes
    - 🏷️ Rich metadata display

    **Usage Tips:**
    - Use clinical terminology
    - Be specific (e.g., "type 2 diabetes" not just "diabetes")
    - Check match scores (🟢 High, 🟡 Medium, 🔴 Low)
    - Select multiple phenotypes to combine codelists
    """)

    st.markdown("---")
    st.markdown("**Examples:**")
    st.code("type 2 diabetes mellitus")
    st.code("myocardial infarction")
    st.code("chronic kidney disease stage 3")
    st.code("asthma in children")

# Main search interface
st.markdown("---")
st.header("🔎 Search Phenotypes")

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., type 2 diabetes, heart failure, chronic obstructive pulmonary disease",
        help="Describe the phenotype you're looking for in natural language",
        label_visibility="collapsed"
    )

with col2:
    max_results = st.number_input("Max results", min_value=5, max_value=50, value=20, step=5)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)

with col2:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state.search_results = []
        st.session_state.selected_phenotypes = {}
        st.session_state.last_query = ""
        st.session_state.expanded_phenotypes = set()
        st.rerun()

# Perform search
if search_button and query:
    st.session_state.last_query = query

    with st.spinner(f"🔍 Searching for: **{query}**"):
        filters = {}
        if phenotype_type_filter != "All":
            filters['phenotype_type'] = phenotype_type_filter

        results = search_phenotypes(query, client, model, filters)
        st.session_state.search_results = results[:max_results]
        st.session_state.selected_phenotypes = {}
        st.session_state.expanded_phenotypes = set()

# Display results
if st.session_state.search_results:
    st.markdown("---")

    # Results header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader(f"📊 Found {len(st.session_state.search_results)} results")
    with col2:
        if st.button("☑️ Select All", use_container_width=True):
            for r in st.session_state.search_results:
                if isinstance(r, dict) and 'phenotype_id' in r and 'name' in r:
                    st.session_state.selected_phenotypes[r['phenotype_id']] = r['name']
            st.rerun()
    with col3:
        if st.button("⬜ Clear Selection", use_container_width=True):
            st.session_state.selected_phenotypes = {}
            st.rerun()

    # Display results
    for idx, result in enumerate(st.session_state.search_results):
        display_phenotype_card(result, idx)

    # Combine selected phenotypes
    if st.session_state.selected_phenotypes:
        st.markdown("---")
        st.header("🔗 Combine Selected Phenotypes")

        st.info(f"✅ **{len(st.session_state.selected_phenotypes)} phenotype(s) selected**")

        # Show selected
        selected_names = list(st.session_state.selected_phenotypes.values())
        with st.expander("View selected phenotypes"):
            for pid, pname in st.session_state.selected_phenotypes.items():
                st.markdown(f"- **{pname}** ({pid})")

        if st.button("🔗 Combine Codelists", type="primary", use_container_width=True):
            with st.spinner("Combining codelists..."):
                combined_codes = []
                progress_bar = st.progress(0)

                for i, (pid, pname) in enumerate(st.session_state.selected_phenotypes.items()):
                    codelist = get_phenotype_codelist(client, pid)
                    if codelist is not None and not codelist.empty:
                        codelist['source_phenotype_id'] = pid
                        codelist['source_phenotype_name'] = pname
                        combined_codes.append(codelist)
                    progress_bar.progress((i + 1) / len(st.session_state.selected_phenotypes))

                if combined_codes:
                    combined_df = pd.concat(combined_codes, ignore_index=True)

                    # Remove duplicates based on code
                    if 'code' in combined_df.columns:
                        original_count = len(combined_df)
                        combined_df = combined_df.drop_duplicates(subset=['code'], keep='first')
                        dedup_count = original_count - len(combined_df)

                        st.success(f"✅ Combined {len(combined_df)} unique codes (removed {dedup_count} duplicates)")
                    else:
                        st.success(f"✅ Combined {len(combined_df)} codes")

                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Codes", len(combined_df))
                    with col2:
                        st.metric("Source Phenotypes", len(st.session_state.selected_phenotypes))
                    with col3:
                        if 'coding_system' in combined_df.columns:
                            st.metric("Coding Systems", combined_df['coding_system'].nunique())
                    with col4:
                        if 'source_phenotype_id' in combined_df.columns:
                            st.metric("Unique Sources", combined_df['source_phenotype_id'].nunique())

                    # Preview
                    st.dataframe(combined_df.head(50), use_container_width=True)

                    # Download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download_button(
                        combined_df,
                        f"combined_phenotypes_{timestamp}.csv",
                        "📥 Download Combined Codelist",
                        f"download_combined_{timestamp}"
                    )
                else:
                    st.error("❌ No codes found in selected phenotypes")

elif st.session_state.last_query:
    st.info(f"No results found for: **{st.session_state.last_query}**. Try different search terms.")
else:
    st.info("👆 Enter a search query above to get started")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.875rem;'>
    Data from <a href='https://phenotypes.healthdatagateway.org/' target='_blank'>HDR UK Phenotype Library</a> |
    Powered by <a href='https://github.com/SwanseaUniversityMedical/pyconceptlibraryclient' target='_blank'>pyconceptlibraryclient</a>
</div>
""", unsafe_allow_html=True)
