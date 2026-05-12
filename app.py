import streamlit as st
from modules.reformulator import reformulate_query
from modules.retriever import retrieve_relevant_docs
from modules.reranker import rerank_documents
from modules.generator import generate_answer
from modules.verifier import verify_answer
from modules.reviser import revise_answer

st.set_page_config(page_title="RAG Answer Verification Agent", layout="wide")

st.title("RAG Answer Verification Agent")
st.write(
    "This prototype reformulates the query, retrieves document chunks, re-ranks the strongest evidence, generates an answer, verifies support, and revises risky answers."
)

question = st.text_input("Enter your question:")

if st.button("Run Agent"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Reformulating query..."):
            reformulated_query = reformulate_query(question)

        with st.spinner("Retrieving candidate chunks..."):
            candidate_docs = retrieve_relevant_docs(reformulated_query, top_k=5)

        with st.spinner("Re-ranking strongest evidence..."):
            retrieved_docs = rerank_documents(question, candidate_docs, top_n=2)

        with st.spinner("Generating answer..."):
            answer = generate_answer(question, retrieved_docs)

        with st.spinner("Verifying answer..."):
            verification = verify_answer(question, answer, retrieved_docs)

        claims = verification.get("claims", [])
        overall_risk = verification.get("overall_risk", "Unknown")

        risky_claim_exists = any(
            item.get("support") in ["Partially Supported", "Unsupported"]
            for item in claims
        )

        if risky_claim_exists:
            with st.spinner("Revising answer..."):
                revised_answer = revise_answer(question, answer, verification, retrieved_docs)
        else:
            revised_answer = answer

        st.subheader("Original Question")
        st.write(question)

        st.subheader("Reformulated Query")
        st.write(reformulated_query)

        st.subheader("Candidate Chunks (Before Re-ranking)")
        for doc in candidate_docs:
            st.markdown(
                f"**{doc['title']} - Chunk {doc['chunk_id']}**  \nRetrieval Score: {doc['score']:.4f}"
            )
            st.write(doc["content"])
            st.markdown("---")

        st.subheader("Selected Evidence (After Re-ranking)")
        for doc in retrieved_docs:
            st.markdown(
                f"**{doc['title']} - Chunk {doc['chunk_id']}**  \nRetrieval Score: {doc['score']:.4f}  \nRe-rank Score: {doc.get('rerank_score', 0):.2f}"
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

        st.subheader("Final Revised Answer")
        st.write(revised_answer)