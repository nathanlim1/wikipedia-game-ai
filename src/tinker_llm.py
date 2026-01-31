import os
import tinker
from tinker import types
from dotenv import load_dotenv

load_dotenv()

class TinkerLLM:
    def __init__(self, model: str = "Qwen/Qwen3-8B"):
        """
        Initialize the Tinker LLM client with a given model. Default model is "Qwen/Qwen3-8B".
        """
        self.model = model
        self.api_key = os.getenv("TINKER_API_KEY")
        if not self.api_key:
            raise ValueError("Tinker API key is required. Set TINKER_API_KEY env var.")
        
        # ServiceClient implicitly uses TINKER_API_KEY env var when set
        self.service_client = tinker.ServiceClient()
        
        print(f"Initializing Tinker SamplingClient with model: {self.model}...")
        self.sampling_client = self.service_client.create_sampling_client(base_model=self.model)
        self.tokenizer = self.sampling_client.get_tokenizer()
        print("Tinker SamplingClient initialized.")

    def chat(self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate a chat completion.

        Args:
            messages (list[dict]): List of message dicts with 'role' and 'content'.
            max_tokens (int): Maximum tokens to generate.
            temperature (float): Sampling temperature.

        Returns:
            str: The generated assistant message content.
        """
        try:
            # add_generation_prompt=True ensures the model generates the assistant response
            text_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            raise ValueError(f"Failed to apply chat template: {e}")

        # Encode inputs using required tokenizer for model
        tokens = self.tokenizer.encode(text_input)
        prompt = types.ModelInput.from_ints(tokens)
        
        # Set up sampling parameters
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            # Can add stop conditions if needed (basedo n docs)
        )
        
        # num_samples=1 for standard chat
        future = self.sampling_client.sample(
            prompt, 
            num_samples=1, 
            sampling_params=params
        )
        
        # Block and get result
        result = future.result()
        
        if not result.sequences:
            return ""
            
        # Decode the first sequence
        output_text = self.tokenizer.decode(result.sequences[0].tokens)
        return output_text

    def completion(self, prompt_text: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """
        Generate a text completion (non-chat).
        """
        tokens = self.tokenizer.encode(prompt_text)
        prompt = types.ModelInput.from_ints(tokens)
        
        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        future = self.sampling_client.sample(
            prompt, 
            num_samples=1, 
            sampling_params=params
        )
        
        result = future.result()
        if not result.sequences:
            return ""
            
        return self.tokenizer.decode(result.sequences[0].tokens)
