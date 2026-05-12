import streamlit as st
from modules.retriever import retrieve_relevant_docs
from modules.reranker import rerank_documents
from modules.cloud_demo_agent import run_cloud_demo_agent

st.set_page_config(page_title="RAG Hallucination Agent", layout="wide")

st.title("RAG Hallucination / Faithfulness Evaluation Agent")
st.write(
    "This cloud demo retrieves relevant document chunks, selects the strongest evidence, "
    "generates an answer, and evaluates whether the answer is supported by the sources."
)

st.info(
    "Cloud demo mode is enabled to reduce API usage. "
    "Generation and verification are handled in a single model call."
)

question = st.text_input("Enter your question:")

if st.button("Run Agent"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving candidate chunks..."):
            candidate_docs = retrieve_relevant_docs(question, top_k=5)

        with st.spinner("Re-ranking strongest evidence..."):
            selected_docs = rerank_documents(question, candidate_docs, top_n=2)

        with st.spinner("Generating and verifying answer..."):
            result = run_cloud_demo_agent(question, selected_docs)

        answer = result.get("answer", "")
        claims = result.get("claims", [])
        overall_risk = result.get("overall_risk", "Unknown")

        st.subheader("Original Question")
        st.write(question)

        st.subheader("Candidate Chunks")
        for doc in candidate_docs:
            st.markdown(
                f"**{doc['title']} - Chunk {doc['chunk_id']}**  \n"
                f"Retrieval Score: {doc['score']:.4f}"
            )
            st.write(doc["content"])
            st.markdown("---")

        st.subheader("Selected Evidence")
        for doc in selected_docs:
            st.markdown(
                f"**{doc['title']} - Chunk {doc['chunk_id']}**  \n"
                f"Retrieval Score: {doc['score']:.4f}  \n"
                f"Re-rank Score: {doc.get('rerank_score', 0):.4f}"
            )
            st.write(doc["content"])
            st.markdown("---")

        st.subheader("Generated Answer")
        st.write(answer)

        st.subheader("Verification Result")
        for i, claim_item in enumerate(claims, start=1):
            claim_text = claim_item.get("claim", "")
            support = claim_item.get("support", "")
            reason = claim_item.get("reason", "")

            st.markdown(f"### Claim {i}")
            st.write(claim_text)

            if support == "Supported":
                st.success(f"Support: {support}")
            elif support == "Partially Supported":
                st.warning(f"Support: {support}")
            elif support == "Unsupported":
                st.error(f"Support: {support}")
            else:
                st.info(f"Support: {support}")

            st.write(f"Reason: {reason}")
            st.markdown("---")

        st.subheader("Overall Risk")
        if overall_risk == "Low":
            st.success(f"Overall Risk: {overall_risk}")
        elif overall_risk == "Medium":
            st.warning(f"Overall Risk: {overall_risk}")
        elif overall_risk == "High":
            st.error(f"Overall Risk: {overall_risk}")
        else:
            st.info(f"Overall Risk: {overall_risk}")
