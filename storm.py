import logging
import json
import os
import re
from pydantic import BaseModel
import dspy
from utils import fetch_wikipedia_links, fetch_table_of_contents

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

class LinkData(BaseModel):
    links: list[str]
    def to_json(self):
        return json.dumps(self.links)

class TableOfContents(BaseModel):
    sections: list[str]
    def to_json(self):
        return json.dumps(self.sections)

class ConversationSignature(dspy.Signature):
    topic = dspy.InputField(desc="Main topic")
    perspective = dspy.InputField(desc="Perspective for the conversation")
    conversation_history = dspy.InputField(desc="Previous conversation history", optional=True)
    question = dspy.OutputField(desc="Generated question")
    answer = dspy.OutputField(desc="Synthesized answer")

class ResearchSignature(dspy.Signature):
    topic: str = dspy.InputField(desc="The topic to research")
    related_topics: str = dspy.OutputField(desc="Wikipedia links related to the topic")
    table_of_contents: str = dspy.OutputField(desc="Table of contents for each related topic")

class GenerateTableOfContentsSignature(dspy.Signature):
    topic: str = dspy.InputField(desc="The main topic")
    related_topics: str = dspy.InputField(desc="Related topics and subtopics")
    rationale: str = dspy.InputField(desc="Rationale for generating the table of contents")
    table_of_contents: str = dspy.OutputField(desc="Generated table of contents")

class PerspectiveSignature(dspy.Signature):
    topic = dspy.InputField(desc="The main topic for which perspectives are needed")
    perspectives = dspy.OutputField(desc="Generated list of perspectives")

class CombinedSignature(dspy.Signature):
    topic = dspy.InputField(desc="Main topic for outline creation")
    content = dspy.InputField(desc="Content gathered from conversations")
    prompt = dspy.InputField(desc="Prompt for generating the article")
    full_article = dspy.OutputField(desc="Completed article text")
    
class FullArticleCreationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_article = dspy.ChainOfThought(CombinedSignature)

    def generate_full_article(self, topic, conversation_history, prompt):
        """
        Generate article iteratively, section by section to avoid repetition.
        Each iteration writes a NEW section, not repeating previous content.
        """
        content = " ".join([answer for _, answer in conversation_history])
        full_article = ""
        target_token_length = 800
        min_section_length = 100  # Minimum words per section
        max_iterations = 10
        iterations = 0

        # Track what sections have been covered
        sections_written = []

        logging.info(f"Starting article generation with target length: {target_token_length} words")

        while len(full_article.split()) < target_token_length and iterations < max_iterations:
            iterations += 1
            current_length = len(full_article.split())
            remaining_words = target_token_length - current_length

            logging.info(f"Generation iteration {iterations}/{max_iterations}, current length: {current_length}/{target_token_length} words")

            # Build directive for this iteration to write NEW content
            if iterations == 1:
                section_prompt = f"""Write the INTRODUCTION section for '{topic}'.
- Provide background and context
- Define key terms
- Preview what the article will cover
Target: {min(remaining_words, 200)} words."""

            elif iterations == 2:
                section_prompt = f"""Write the MAIN TECHNOLOGIES/METHODS section for '{topic}'.
- Focus on HOW it works, technical details, mechanisms
- Include specific examples and technologies
- DO NOT repeat benefits/challenges from intro
- DO NOT redefine terms already covered
Target: {min(remaining_words, 250)} words."""

            elif iterations == 3:
                section_prompt = f"""Write the REAL-WORLD APPLICATIONS section for '{topic}'.
- Focus on concrete examples, case studies, current uses
- Industries, companies, projects using this
- Specific numerical data if available
- DO NOT repeat general concepts already covered
Target: {min(remaining_words, 250)} words."""

            elif iterations == 4:
                section_prompt = f"""Write the CHALLENGES AND FUTURE section for '{topic}'.
- Technical/practical obstacles
- Current research directions
- What's on the horizon
- DO NOT re-list benefits or basic applications already mentioned
Target: {min(remaining_words, 200)} words."""

            else:
                section_prompt = f"""Write a brief CONCLUSION for '{topic}'.
- Synthesize the big picture
- Future outlook
- DO NOT repeat specific details, examples, or lists from earlier sections
- Keep it high-level and forward-looking
Target: {min(remaining_words, 150)} words."""

            # Include what's already written so LLM knows what NOT to repeat
            if full_article:
                section_prompt += f"\n\n=== CRITICAL: DO NOT REPEAT ANY OF THIS ===\n{full_article[-600:]}"  # Show more context

            logging.info(f"  Section directive: {section_prompt[:100]}...")

            # Generate new section
            prediction = self.process_article(
                topic=topic,
                content=content,
                prompt=section_prompt
            )

            if hasattr(prediction, 'full_article'):
                generated_text = prediction.full_article.strip()
                word_count = len(generated_text.split())

                if word_count < 50:
                    logging.warning(f"Generated section too short ({word_count} words). Skipping.")
                    continue

                # Add the new section
                full_article += "\n\n" + generated_text
                sections_written.append(f"Iteration {iterations}: {word_count} words")

                logging.info(f"  ✓ Added section: {word_count} words")

                if len(full_article.split()) >= target_token_length:
                    logging.info(f"Target length reached!")
                    break
            else:
                logging.error("Failed to generate a segment.")
                break

        final_word_count = len(full_article.split())
        logging.info(f"Article generation complete. Final length: {final_word_count} words after {iterations} iterations")
        logging.info(f"Sections written: {sections_written}")

        return full_article.strip()

class ResearchAndConversationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research_module = dspy.ChainOfThought(ResearchSignature)
        self.generate_toc_module = dspy.ChainOfThought(GenerateTableOfContentsSignature)
        self.conversation_module = dspy.ChainOfThought(ConversationSignature)
        self.perspective_predict = dspy.Predict(PerspectiveSignature)
        self.article_module = FullArticleCreationModule()

    def forward(self, topic):
        related_topics = fetch_wikipedia_links(topic)
        toc_data = self.generate_toc_module(topic=topic, related_topics=LinkData(links=related_topics).to_json(), rationale="Generate detailed TOC")
        table_of_contents = toc_data.table_of_contents if hasattr(toc_data, 'table_of_contents') else "No TOC generated"

        perspectives_output = self.perspective_predict(topic=topic)
        conversation_history = [("Initial query", f"Introduction to {topic}")]
        formatted_history = ' '.join([f"{q}: {a}" for q, a in conversation_history])
        conversation_output = self.conversation_module(topic=topic, perspective=perspectives_output.get('perspectives', ''), conversation_history=formatted_history)
        updated_history = conversation_history + [(conversation_output.question, conversation_output.answer)]
        prompt = "The impact of sustainable energy on global economies"
        generated_article = self.article_module.generate_full_article(topic, updated_history, prompt)

        return {
            "research": {"related_topics": related_topics, "table_of_contents": table_of_contents},
            "conversation": {"next_question": conversation_output.question, "answer": conversation_output.answer, "history": updated_history},
            "perspectives": perspectives_output.perspectives.split("\n") if 'perspectives' in perspectives_output else [],
            "article": generated_article
        }

if __name__ == "__main__":
    import sys
    module = ResearchAndConversationModule()
    topic = sys.argv[1] if len(sys.argv) > 1 else "Sustainable Energy"
    results = module.forward(topic)

    print("\n" + "="*80)
    print(f"STORM RESULTS: {topic}")
    print("="*80)
    print(f"Word Count: {len(results['article'].split())} words")
    print(f"Perspectives: {len(results['perspectives'])}")
    print("="*80 + "\n")

    print("ARTICLE:")
    print("="*80)
    print(results['article'])
    print("\n" + "="*80)
    print("Full JSON results:")
    print(json.dumps(results, indent=4))
