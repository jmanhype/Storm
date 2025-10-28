import os
import dspy
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get OpenRouter API key from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not found")

# Initialize DSPy settings with OpenRouter
lm = dspy.LM(
    model="openrouter/anthropic/claude-3-haiku",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1"
)
dspy.settings.configure(lm=lm)

class CombinedSignature(dspy.Signature):
    topic = dspy.InputField(desc="Main topic for outline creation")
    content = dspy.InputField(desc="Content gathered from conversations")
    prompt = dspy.InputField(desc="Prompt for generating the article")
    full_article = dspy.OutputField(desc="Completed article text")

def parse_outline(text):
    """Attempts to parse narrative text into sections based on detected headers or paragraph breaks."""
    outline_dict = {}
    current_header = None
    content_list = []
    sections = text.split('\n')
    for line in sections:
        if re.match(r"^\d+\.\s", line):
            if current_header:
                outline_dict[current_header] = ' '.join(content_list).strip()
            current_header = line.strip()
            content_list = []
        else:
            content_list.append(line.strip())
    if current_header and content_list:
        outline_dict[current_header] = ' '.join(content_list).strip()
    return outline_dict if outline_dict else {'Full Article': text}

class OutlineCreationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_article = dspy.ChainOfThought(CombinedSignature)

    def forward(self, topic, conversation_history):
        content = " ".join([answer for _, answer in conversation_history])
        prompt = f"Create an outline for the topic: {topic}"
        prediction = self.process_article(topic=topic, content=content, prompt=prompt)
        if hasattr(prediction, 'full_article'):
            return prediction.full_article
        else:
            logging.error("Failed to generate outline.")
            return None

class FullArticleCreationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_article = dspy.ChainOfThought(CombinedSignature)

    def generate_full_article(self, topic, conversation_history, prompt):
        content = " ".join([answer for _, answer in conversation_history])
        full_article = ""
        target_token_length = 800  # Increased target token length for longer articles
        min_paragraph_length = 65  # Minimum number of words per paragraph
        
        while len(full_article.split()) < target_token_length:
            prediction = self.process_article(topic=topic, content=content, prompt=prompt)
            if hasattr(prediction, 'full_article'):
                generated_text = prediction.full_article.strip()
                paragraphs = generated_text.split('\n')
                
                for paragraph in paragraphs:
                    if len(paragraph.split()) >= min_paragraph_length:
                        full_article += "\n\n" + paragraph
                    else:
                        prompt += " " + paragraph  # Append the short paragraph to the prompt for further generation
                
                if len(full_article.split()) >= target_token_length:
                    break  # Stop generation if we reach or exceed the target token length
            else:
                logging.error("Failed to generate a segment.")
                break
        
        return full_article.strip()

# Example of using the module
if __name__ == "__main__":
    article_module = FullArticleCreationModule()
    topic = "Sustainable Energy"
    conversation_history = [
        ("What is renewable energy?", "Renewable energy sources are naturally replenishing."),
        ("Why is it important?", "It's important because it has a lower environmental impact and is sustainable.")
    ]
    prompt = "The impact of renewable energy on global economies"
    generated_article = article_module.generate_full_article(topic, conversation_history, prompt)
    print("Generated Article:", generated_article)