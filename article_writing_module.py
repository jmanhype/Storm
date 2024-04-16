import dspy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize DSPy settings with a large language model
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="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")
dspy.settings.configure(lm=claude)

class ArticleWritingSignature(dspy.Signature):
    outline = dspy.InputField(desc="Final article outline")
    full_article = dspy.OutputField(desc="Completed article text")

class ArticleWritingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.article_predict = dspy.ChainOfThought(ArticleWritingSignature)

    def forward(self, outline, references):
        sections = []
        for section_title, content in outline.items():
            try:
                section_text = self.article_predict(outline=section_title, full_article=content)
                if hasattr(section_text, 'full_article') and section_text.full_article:
                    sections.append(section_text.full_article)
                else:
                    logging.warning(f"No content generated for section: {section_title}")
                    sections.append(f"Default content for {section_title}")
            except Exception as e:
                logging.error(f"Error generating content for section {section_title}: {str(e)}")
                sections.append(f"Error content for {section_title}")

        full_text = " ".join(sections)
        try:
            final_article = self.article_predict(outline="Complete Article", full_article=full_text)
            return final_article.full_article if hasattr(final_article, 'full_article') else "Failed to generate the final article."
        except Exception as e:
            logging.error(f"Error generating the final article: {str(e)}")
            return "Failed to generate the final article."


if __name__ == "__main__":
    article_module = ArticleWritingModule()
    example_outline = {
        "Introduction": "Introduction to sustainable energy",
        "Main Body": "Detailed discussion on solar and wind energy",
        "Conclusion": "The future of renewable energy"
    }
    example_references = {
        "Introduction": "Sustainable energy is important for global development.",
        "Main Body": "Solar energy harnesses the sun's power; wind energy harnesses wind power.",
        "Conclusion": "Renewable energy will play a crucial role in future energy solutions."
    }
    result = article_module.forward(example_outline, example_references)
    print("Generated Article:", result)
