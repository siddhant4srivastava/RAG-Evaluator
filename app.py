import os
import asyncio

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import get_llm, get_embeddings, PROVIDER


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

CHROMA_PATH = "chroma_db"


PROMPT_TEMPLATE = """Use only the following context to answer the question.

If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Evaluator",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title("🔍 RAG Evaluator")

st.markdown(
    """
    **End-to-end Retrieval-Augmented Generation with RAGAS evaluation**

    This demo allows you to inspect:
    - Retrieved context
    - Generated answer
    - RAGAS evaluation metrics
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.write(
        f"**LLM Provider:** `{PROVIDER.upper()}`"
    )

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.markdown("### Evaluation Metrics")

    st.markdown(
        """
        **Faithfulness**  
        Is the answer grounded in the retrieved context?

        **Answer Relevancy**  
        Does the answer address the question?

        **Context Precision**  
        Are the retrieved chunks relevant?

        **Context Recall**  
        Was the required information retrieved?
        """
    )


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():

    if not os.path.exists(CHROMA_PATH):

        raise FileNotFoundError(
            "ChromaDB was not found. "
            "Please run `python ingest.py` first."
        )

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    return vectorstore


# ============================================================
# LOAD LLM
# ============================================================

@st.cache_resource
def load_llm():

    return get_llm()


# ============================================================
# RAG FUNCTION
# ============================================================

def run_rag(
    question,
    top_k
):

    vectorstore = load_vectorstore()

    llm = load_llm()

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    docs = vectorstore.similarity_search(
        question,
        k=top_k
    )

    contexts = [
        doc.page_content
        for doc in docs
    ]

    # --------------------------------------------------------
    # FORMAT CONTEXT
    # --------------------------------------------------------

    context = "\n\n".join(contexts)

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    # --------------------------------------------------------
    # GENERATION
    # --------------------------------------------------------

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return answer.strip(), contexts


# ============================================================
# RAGAS EVALUATION
# ============================================================

def evaluate_with_ragas(
    question,
    answer,
    contexts,
    ground_truth
):

    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall
    )

    from ragas.llms import llm_factory
    from ragas.embeddings import embedding_factory

    from openai import AsyncOpenAI

    if PROVIDER != "openai":

        raise ValueError(
            "The Streamlit RAGAS demo currently uses "
            "the OpenAI evaluator. Set PROVIDER=openai."
        )

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client
    )

    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    faithfulness = Faithfulness(
        llm=evaluator_llm
    )

    answer_relevancy = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    context_precision = ContextPrecision(
        llm=evaluator_llm
    )

    context_recall = ContextRecall(
        llm=evaluator_llm
    )

    # --------------------------------------------------------
    # RAGAS 0.4 scoring
    # --------------------------------------------------------

    async def score():

        faithfulness_result = await faithfulness.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        relevancy_result = await answer_relevancy.ascore(
            user_input=question,
            response=answer
        )

        precision_result = await context_precision.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth
        )

        recall_result = await context_recall.ascore(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth
        )

        return {
            "faithfulness": float(
                faithfulness_result.value
            ),
            "answer_relevancy": float(
                relevancy_result.value
            ),
            "context_precision": float(
                precision_result.value
            ),
            "context_recall": float(
                recall_result.value
            )
        }

    return asyncio.run(score())


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask your Knowledge Base")

question = st.text_input(
    "Enter your question",
    placeholder="Example: What is machine learning?"
)


# ============================================================
# RUN BUTTON
# ============================================================

if st.button(
    "🚀 Run RAG Pipeline",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    try:

        with st.spinner(
            "Retrieving context and generating answer..."
        ):

            answer, contexts = run_rag(
                question,
                top_k
            )

        # Save to session
        st.session_state["question"] = question
        st.session_state["answer"] = answer
        st.session_state["contexts"] = contexts

    except Exception as e:

        st.error(
            f"RAG pipeline failed: {str(e)}"
        )

        st.stop()


# ============================================================
# DISPLAY ANSWER
# ============================================================

if "answer" in st.session_state:

    st.divider()

    st.subheader("❓ Question")

    st.info(
        st.session_state["question"]
    )

    st.subheader("🤖 Generated Answer")

    st.success(
        st.session_state["answer"]
    )

    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    st.subheader(
        "📚 Retrieved Context"
    )

    contexts = st.session_state["contexts"]

    for i, context in enumerate(contexts):

        with st.expander(
            f"Chunk {i + 1}"
        ):

            st.write(context)


    # ========================================================
    # EVALUATION
    # ========================================================

    st.divider()

    st.subheader(
        "📊 RAGAS Evaluation"
    )

    ground_truth = st.text_area(
        "Ground Truth Answer",
        placeholder=(
            "Enter the expected answer for this question."
        )
    )

    if st.button(
        "📈 Evaluate with RAGAS"
    ):

        if not ground_truth.strip():

            st.warning(
                "Please provide a ground truth answer."
            )

            st.stop()

        try:

            with st.spinner(
                "Running RAGAS evaluation..."
            ):

                scores = evaluate_with_ragas(
                    question=st.session_state["question"],
                    answer=st.session_state["answer"],
                    contexts=st.session_state["contexts"],
                    ground_truth=ground_truth
                )

            # ------------------------------------------------
            # DISPLAY SCORES
            # ------------------------------------------------

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Faithfulness",
                f"{scores['faithfulness']:.3f}"
            )

            col2.metric(
                "Answer Relevancy",
                f"{scores['answer_relevancy']:.3f}"
            )

            col3.metric(
                "Context Precision",
                f"{scores['context_precision']:.3f}"
            )

            col4.metric(
                "Context Recall",
                f"{scores['context_recall']:.3f}"
            )

            # ------------------------------------------------
            # INTERPRETATION
            # ------------------------------------------------

            st.subheader(
                "🧠 Interpretation"
            )

            if scores["faithfulness"] < 0.7:

                st.warning(
                    "Low Faithfulness → possible hallucination "
                    "or unsupported claims."
                )

            else:

                st.success(
                    "High Faithfulness → answer is well grounded "
                    "in retrieved context."
                )


            if scores["context_precision"] < 0.7:

                st.warning(
                    "Low Context Precision → retriever may be "
                    "returning noisy or irrelevant chunks."
                )


            if scores["context_recall"] < 0.7:

                st.warning(
                    "Low Context Recall → retriever may be "
                    "missing information required to answer."
                )

        except Exception as e:

            st.error(
                f"RAGAS evaluation failed: {str(e)}"
            )