from llama_cpp import Llama
import logging

class LocalGenerator:
    """
    A wrapper class to handle loading and running a local GGUF model
    using llama-cpp-python for text generation.
    """
    def __init__(self, model_path: str, n_gpu_layers: int, n_ctx: int):
        print("Initializing local LLM...")
        try:
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=True,  # Set to False for less console output
                chat_format="llama-3" # Use the correct chat format for your model
            )
            print("✅ Local LLM loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load local LLM. Error: {e}")
            print("Please ensure you have the correct model path and that llama-cpp-python is installed correctly.")
            self.llm = None

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        """
        Generates a response from the local LLM based on the given prompt.
        """
        if not self.llm:
            return "Error: Local LLM is not available."

        # We will use the chat completions endpoint which is standard
        # The prompt is wrapped in a user message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|eot_id|>", "<|end_of_text|>"] # Llama 3 specific stop tokens
            )
            content = response['choices'][0]['message']['content']
            return content.strip()
        except Exception as e:
            logging.error(f"Error during LLM generation: {e}", exc_info=True)
            return "Error occurred during text generation."