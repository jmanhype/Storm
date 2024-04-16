import dspy
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize DSPy settings with a large language model
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="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")
dspy.settings.configure(lm=claude)

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