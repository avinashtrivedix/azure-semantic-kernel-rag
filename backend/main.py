import os
import asyncio
from dotenv import load_dotenv

# Import Semantic Kernel core components
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.ollama import OllamaChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig

# Load the variables from the .env file
load_dotenv()

async def main():
    # 1. Initialize the Kernel (The "Motherboard")
    kernel = Kernel()

    # 2. Check our architectural switch
    provider = os.getenv("AI_PROVIDER", "ollama").lower()

    if provider == "azure":
        print("🟢 Routing to Enterprise: Azure OpenAI")
        # Plug Azure into the motherboard
        chat_service = AzureChatCompletion(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            service_id="ai_chat"
        )
    else:
        print("🔵 Routing to Local: Ollama")
        # Plug Local AI into the motherboard
        chat_service = OllamaChatCompletion(
            ai_model_id=os.getenv("OLLAMA_MODEL"),
            host=os.getenv("OLLAMA_ENDPOINT"),
            service_id="ai_chat"
        )

    # Register the chosen AI service to the kernel
    kernel.add_service(chat_service)

    # 3. Test the connection
    print("System online. Asking the AI a test question...")
    
    prompt = "In one sentence, what is Semantic Kernel?"
    response = await kernel.invoke_prompt(prompt)
    
    print(f"\nAI Response: {response}")

# Run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(main())