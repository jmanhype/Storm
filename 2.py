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
    """Parses the structured text into a dictionary for easier processing."""
    outline_dict = {}
    current_section = None
    content_list = []

    for line in text.split('\n'):
        if re.match(r'^\d+\.', line.strip()):
            if current_section and content_list:
                outline_dict[current_section] = ' '.join(content_list)
                content_list = []
            current_section = line.strip()
        else:
            content_list.append(line.strip())

    if current_section and content_list:
        outline_dict[current_section] = ' '.join(content_list)

    return outline_dict

class FullArticleCreationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_article = dspy.ChainOfThought(CombinedSignature)

    def generate_full_article(self, topic, conversation_history, prompt):
        content = " ".join([answer for _, answer in conversation_history])
        prediction = self.process_article(topic=topic, content=content, prompt=prompt)
        logging.info(f"Outline response: {prediction}")

        if prediction and hasattr(prediction, 'outline'):
            outline = parse_outline(prediction.outline)
            logging.info(f"Parsed outline: {outline}")

            if not outline:
                logging.error("Failed to parse outline.")
                return "Failed to generate the article due to outline parsing issues."

            sections = []
            for section_title, content in outline.items():
                segment = self.process_article(prompt=content)
                if hasattr(segment, 'full_article') and segment.full_article:
                    sections.append(segment.full_article)
                else:
                    logging.warning(f"No content generated for section: {section_title}")
                    sections.append(f"Content not generated for section: {section_title}")

            full_article = " ".join(sections)
            return full_article

        return "Failed to generate the article due to missing outline."

if __name__ == "__main__":
    article_module = FullArticleCreationModule()
    topic = "Sustainable Energy"
    conversation_history = [
        ("What is renewable energy?", "Renewable energy is energy from sources that are naturally replenishing."),
        ("Why is it important?", "It's important because it has a lower environmental impact and is sustainable.")
    ]
    prompt = "The impact of renewable energy on global economies"

    generated_article = article_module.generate_full_article(topic, conversation_history, prompt)
    print("Generated Article:", generated_article)
