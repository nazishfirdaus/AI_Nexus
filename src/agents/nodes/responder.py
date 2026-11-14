import logfire
from src.agents.state import AgentState
from src.gateway import llm_client


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Groq client (not LangChain) for the final synthesis.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful loan-document assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating loan-document RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a senior loan-underwriting analyst.
        Answer the question using the LOAN DOCUMENT CONTEXT provided.

        LOAN DOCUMENT CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"

        Rules:
        - Use only the loan document context and conversation history.
        - Quote amounts, dates, account numbers, and names exactly as they appear.
        - Cite the source document and page for every factual claim.
        - If the documents do not contain the answer, say so explicitly; do not guess.
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content

            logfire.info("✅ Response synthesised.")

            return {
                "final_answer": content,
                "status": "Response generated.",
                "plan": state["plan"],
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e