import os
import dspy
import logging

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

class ArticleWritingSignature(dspy.Signature):
    outline = dspy.InputField(desc="Final article outline")
    full_article = dspy.OutputField(desc="Completed article text")

class ArticleWritingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.article_predict = dspy.ChainOfThought(ArticleWritingSignature)

    def forward(self, outline, references):
        # Handle both string and dict inputs
        if isinstance(outline, str):
            # If outline is a string, use it directly
            try:
                final_article = self.article_predict(outline=outline, full_article="")
                return final_article.full_article if hasattr(final_article, 'full_article') else "Failed to generate the final article."
            except Exception as e:
                logging.error(f"Error generating the final article: {str(e)}")
                return "Failed to generate the final article."

        # Handle dict outline (legacy support)
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
