import torch


from rich import print
from utils import ascii_art
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class TextGenerator:
    def __init__(self, 
                 model: AutoModelForCausalLM, 
                 tokenizer: AutoTokenizer):

        self.conversation_history = []
        self.model = model
        self.tokenizer = tokenizer

    def unload_model(self):
        """
        Unloads the model and tokenizer to free up resources.
        """
        # Clear model from GPU memory
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()  # Clear cached memory

        # Clear tokenizer
        if self.tokenizer is not None:
            del self.tokenizer

        # Optionally, you can also set these to None to break any references
        self.model = None
        self.tokenizer = None

    def talk_to(self):
        self.model.eval()

        while True:
            print("[bold yellow] user : [/bold yellow]")
            eval_prompt = input("")

            if eval_prompt == "!bye":
                self.unload_model()
                break
            elif eval_prompt == "!history":
                print(self.conversation_history)
            elif eval_prompt =="!window":
                window = self.conversation_history[-6:]
                if window[0]["role"] == 'assistant':
                    window = window[1:]
                print(window)
            else:
                self.conversation_history.append({"role": "user", "content": eval_prompt})
                window = self.conversation_history[-6:]
                if window[0]["role"] == 'assistant':
                    window = window[1:]

                model_input = self.tokenizer.apply_chat_template(window, return_tensors="pt").to("cuda")

                num_input_tokens = model_input.shape[1]

                streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

                print("[bold green]assistant : [/bold green]")

                generated_tokens = self.model.generate(model_input, 
                                                    streamer=streamer, 
                                                    max_new_tokens=100, 
                                                    repetition_penalty=1.15)

                output_tokens = generated_tokens[:, num_input_tokens:]
                assistant_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

                # Add assistant response to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_response})