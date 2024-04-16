import dspy
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize DSPy settings with a large language model
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="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")
dspy.settings.configure(lm=claude)

class OutlineCreationSignature(dspy.Signature):
    topic = dspy.InputField(desc="Main topic")
    content = dspy.InputField(desc="Content gathered from conversations")
    outline = dspy.OutputField(desc="Drafted article outline")

class ArticleWritingSignature(dspy.Signature):
    outline = dspy.InputField(desc="Final article outline")
    full_article = dspy.OutputField(desc="Completed article text")

def parse_outline(text):
    """Parses the structured text into a dictionary."""
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

class CombinedModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.outline_predict = dspy.ChainOfThought(OutlineCreationSignature)
        self.article_predict = dspy.ChainOfThought(ArticleWritingSignature)

    def create_outline(self, topic, conversation_history):
        content = " ".join([answer for _, answer in conversation_history])
        prediction = self.outline_predict(topic=topic, content=content)
        if prediction and hasattr(prediction, 'outline'):
            try:
                outline_dict = parse_outline(prediction.outline)
                return outline_dict
            except Exception as e:
                logging.error(f"Failed to parse outline: {str(e)}")
                return None
        else:
            return None

    def write_article(self, outline, references):
        sections = []
        if isinstance(outline, dict):
            for section_title, content in outline.items():
                section_text = self.article_predict(outline=section_title, full_article=content)
                if hasattr(section_text, 'full_article') and section_text.full_article:
                    sections.append(section_text.full_article)
                else:
                    logging.warning(f"No content generated for section: {section_title}")
            full_text = " ".join(sections)
            final_article = self.article_predict(outline="Complete Article", full_article=full_text)
            return final_article.full_article if hasattr(final_article, 'full_article') else "Failed to generate the final article."
        else:
            logging.error("Invalid outline format. Expected a dictionary.")
            return "Failed to generate the final article due to invalid outline format."

if __name__ == "__main__":
    combined_module = CombinedModule()
    example_topic = "Sustainable Energy"
    example_history = [
        ("What is renewable energy?", "Renewable energy is energy from sources that are naturally replenishing."),
        ("Why is it important?", "It's important because it has a lower environmental impact and is sustainable.")
    ]
    
    outline = combined_module.create_outline(example_topic, example_history)
    if outline:
        print("Generated Outline:", outline)
        example_references = {
            "Introduction": "Sustainable energy is important for global development.",
            "Main Body": "Solar energy harnesses the sun's power; wind energy harnesses wind power.",
            "Conclusion": "Renewable energy will play a crucial role in future energy solutions."
        }
        article = combined_module.write_article(outline, example_references)
        print("Generated Article:", article)
    else:
        print("Failed to generate an outline.")
