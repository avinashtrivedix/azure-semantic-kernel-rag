import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion, OllamaTextEmbedding
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore

# Load the .env variables
load_dotenv()

async def main():
    print("1. Initializing Semantic Kernel (The Motherboard)...")
    kernel = Kernel()

    # Add the Chat Model (The Generator)
    chat_service = OllamaChatCompletion(
        ai_model_id=os.getenv("OLLAMA_MODEL"),
        host=os.getenv("OLLAMA_ENDPOINT"),
        service_id="chat"
    )
    kernel.add_service(chat_service)

    # Add the Embedding Model (The Vectorizer)
    embedding_service = OllamaTextEmbedding(
        ai_model_id=os.getenv("OLLAMA_EMBEDDING_MODEL"),
        host=os.getenv("OLLAMA_ENDPOINT"),
        service_id="embedding"
    )
    kernel.add_service(embedding_service)

    # Create an in-memory Vector Database
    memory = SemanticTextMemory(
        storage=VolatileMemoryStore(), 
        embeddings_generator=embedding_service
    )

    print("\n--- PHASE 1: INGESTION ---")
    # Read the text file we created earlier
    # Note: We use ../data/ because we are running this from inside the backend/ folder
    file_path = "../data/hr_policy.txt"
    
    with open(file_path, "r") as f:
        document_text = f.read()
    
    print(f"Reading document: {file_path}")
    # split the documenr into chunks for simplicity by paragraph
    chunks = document_text.split("\n\n")
    
    # Loop through each chunk, vectorize it, and save it individually
    for index, chunk in enumerate(chunks):
        if chunk.strip(): # Ensure we don't save empty blank lines
            print(f"Embedding and storing chunk {index + 1}...")
            await memory.save_information(
                collection="hr_policies",
                id=f"policy_chunk_{index}",
                text=chunk.strip()
            )
            
    print("All chunks successfully vectorized and saved to database.")

    print("\n--- PHASE 2: RETRIEVAL ---")
    # The user asks a question that the base LLM usually hallucinates
    user_query = "What specific information must be scrubbed before sending it to an LLM?"
    print(f"User Query: '{user_query}'")
    
    # Run Cosine Similarity to find the most relevant text chunk
    search_results = await memory.search(
        collection="hr_policies",
        query=user_query,
        limit=1,
        min_relevance_score=0.5
    )
    
    # Extract the retrieved text, or default to an empty string if nothing matched
    retrieved_context = search_results[0].text if search_results else "No relevant information found."
    print(f"Retrieved Context: '{retrieved_context}'")

    print("\n--- PHASE 3: GENERATION (GROUNDING) ---")
    # This is the magic. We lock the AI inside a strict prompt.
    rag_prompt = f"""
    You are a strict compliance AI. Answer the user's question using ONLY the provided context below. 
    If the answer is not contained in the context, you must reply exactly with: "I do not have enough information to answer."

    Context: {retrieved_context}
    
    Question: {user_query}
    """
    
    print("Generating final hallucination-free answer...")
    response = await kernel.invoke_prompt(rag_prompt)
    
    print(f"\n✅ FINAL SYSTEM RESPONSE: {response}")

if __name__ == "__main__":
    asyncio.run(main())