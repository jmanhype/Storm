import logging
import json
import os
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

# From main.py - Better article generation
class ArticleOutlineSignature(dspy.Signature):
    """Creates a structured outline from conversation history"""
    topic = dspy.InputField(desc="Main topic for outline creation")
    conversation_summary = dspy.InputField(desc="Summary of research and conversations")
    table_of_contents = dspy.InputField(desc="Detailed table of contents")
    outline = dspy.OutputField(desc="Structured article outline with key points")

class ArticleWritingSignature(dspy.Signature):
    """Generates a complete, coherent article from outline"""
    topic = dspy.InputField(desc="Article topic")
    outline = dspy.InputField(desc="Article outline with key points")
    conversation_context = dspy.InputField(desc="Supporting context from conversations")
    full_article = dspy.OutputField(desc="Complete, well-structured article of at least 1000 words with detailed explanations, examples, and smooth transitions between sections")

class EnhancedOutlineCreationModule(dspy.Module):
    """Single-pass outline creation using conversation history"""
    def __init__(self):
        super().__init__()
        self.create_outline = dspy.ChainOfThought(ArticleOutlineSignature)

    def forward(self, topic, conversation_history, table_of_contents):
        # Summarize conversations
        conversation_summary = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in conversation_history
        ])

        logging.info(f"Creating outline for: {topic}")
        prediction = self.create_outline(
            topic=topic,
            conversation_summary=conversation_summary,
            table_of_contents=table_of_contents
        )

        if hasattr(prediction, 'outline'):
            logging.info("Outline created successfully")
            return prediction.outline
        else:
            logging.error("Failed to generate outline")
            return f"Outline for {topic}\n\n{conversation_summary}"

class EnhancedArticleWritingModule(dspy.Module):
    """
    Configurable article generation with length control
    Uses prompt engineering instead of loops to avoid repetition
    """
    def __init__(self, target_words=1000, min_words=500):
        super().__init__()
        self.write_article = dspy.ChainOfThought(ArticleWritingSignature)
        self.target_words = target_words
        self.min_words = min_words

    def forward(self, topic, outline, conversation_history):
        # Create context from conversations
        conversation_context = "\n".join([
            f"{q}: {a}" for q, a in conversation_history
        ])

        logging.info(f"Generating article for: {topic} (target: {self.target_words} words, min: {self.min_words} words)")

        # Enhanced prompt with length specification
        enhanced_outline = f"""TARGET LENGTH: {self.target_words} words (minimum {self.min_words} words)

OUTLINE:
{outline}

INSTRUCTIONS:
- Write a comprehensive, detailed article covering all sections in the outline
- Each major section should be 150-200 words
- Include specific examples, explanations, and transitions
- Ensure smooth flow between sections
- Meet or exceed the target word count of {self.target_words} words
"""

        prediction = self.write_article(
            topic=topic,
            outline=enhanced_outline,
            conversation_context=conversation_context
        )

        if hasattr(prediction, 'full_article'):
            article = prediction.full_article.strip()
            word_count = len(article.split())

            if word_count < self.min_words:
                logging.warning(f"Article too short ({word_count} words < {self.min_words} min). Regenerating with emphasis on length...")
                # Try once more with even stronger emphasis
                enhanced_outline += f"\n\nWARNING: Previous attempt was too short. MUST generate AT LEAST {self.min_words} words!"
                prediction = self.write_article(
                    topic=topic,
                    outline=enhanced_outline,
                    conversation_context=conversation_context
                )
                if hasattr(prediction, 'full_article'):
                    article = prediction.full_article.strip()
                    word_count = len(article.split())

            logging.info(f"Article generated successfully: {word_count} words (target: {self.target_words})")
            return article
        else:
            logging.error("Failed to generate article")
            return "Failed to generate the article."

class EnhancedStormModule(dspy.Module):
    """
    Enhanced STORM: Combines storm.py's comprehensive research
    with main.py's clean article generation

    Parameters:
    -----------
    target_words : int
        Target word count for the article (default: 1000)
    min_words : int
        Minimum acceptable word count (default: 500)
    max_perspectives : int
        Maximum number of perspectives to explore (default: 5)
    """
    def __init__(self, target_words=1000, min_words=500, max_perspectives=5):
        super().__init__()
        self.target_words = target_words
        self.min_words = min_words
        self.max_perspectives = max_perspectives

        # Research phase (from storm.py - comprehensive)
        self.research_module = dspy.ChainOfThought(ResearchSignature)
        self.generate_toc_module = dspy.ChainOfThought(GenerateTableOfContentsSignature)
        self.conversation_module = dspy.ChainOfThought(ConversationSignature)
        self.perspective_predict = dspy.Predict(PerspectiveSignature)

        # Article generation phase (from main.py - clean, no repetition)
        self.outline_module = EnhancedOutlineCreationModule()
        self.article_module = EnhancedArticleWritingModule(
            target_words=target_words,
            min_words=min_words
        )

    def forward(self, topic):
        logging.info(f"Starting Enhanced STORM for topic: {topic}")

        # === RESEARCH PHASE (storm.py approach) ===
        logging.info("Phase 1: Research - Gathering information")
        related_topics = fetch_wikipedia_links(topic)

        # Generate comprehensive table of contents
        toc_data = self.generate_toc_module(
            topic=topic,
            related_topics=LinkData(links=related_topics).to_json(),
            rationale="Generate detailed TOC based on key subtopics"
        )
        table_of_contents = toc_data.table_of_contents if hasattr(toc_data, 'table_of_contents') else "No TOC generated"
        logging.info("✓ Table of contents generated")

        # Generate multiple perspectives
        logging.info("Phase 2: Perspectives - Generating viewpoints")
        perspectives_output = self.perspective_predict(topic=topic)
        perspectives_list = perspectives_output.perspectives.split("\n") if hasattr(perspectives_output, 'perspectives') else []
        logging.info(f"✓ {len(perspectives_list)} perspectives generated")

        # Conduct conversations from different perspectives
        logging.info("Phase 3: Conversations - Engaging in Q&A")
        conversation_history = [("Initial query", f"Introduction to {topic}")]

        # Loop through perspectives like main.py does (better coverage)
        perspective_list = perspectives_output.perspectives.split("\n") if hasattr(perspectives_output, 'perspectives') else []
        for i, perspective in enumerate(perspective_list[:self.max_perspectives]):
            if perspective.strip():  # Skip empty lines
                logging.info(f"  Conversation {i+1}/{self.max_perspectives}: {perspective[:50]}...")
                formatted_history = ' '.join([f"{q}: {a}" for q, a in conversation_history])

                conversation_output = self.conversation_module(
                    topic=topic,
                    perspective=perspective,
                    conversation_history=formatted_history
                )

                conversation_history.append((conversation_output.question, conversation_output.answer))

        logging.info(f"✓ Conversations completed: {len(conversation_history)} exchanges")

        # === ARTICLE GENERATION PHASE (main.py approach) ===
        logging.info("Phase 4: Outline Creation - Structuring article")
        outline = self.outline_module.forward(topic, conversation_history, table_of_contents)

        logging.info("Phase 5: Article Writing - Single-pass generation")
        generated_article = self.article_module.forward(topic, outline, conversation_history)

        # Get the last conversation for reporting
        last_question = conversation_history[-1][0] if len(conversation_history) > 1 else ""
        last_answer = conversation_history[-1][1] if len(conversation_history) > 1 else ""

        # Return structured results
        return {
            "research": {
                "related_topics": related_topics,
                "table_of_contents": table_of_contents
            },
            "conversation": {
                "next_question": last_question,
                "answer": last_answer,
                "history": conversation_history
            },
            "perspectives": perspectives_list,
            "outline": outline,
            "article": generated_article,
            "metadata": {
                "word_count": len(generated_article.split()),
                "num_perspectives": len(perspectives_list),
                "num_conversations": len(conversation_history)
            }
        }

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced STORM: Comprehensive research + clean article generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default settings (1000 words, 5 perspectives)
  python storm_enhanced.py "Quantum Computing"

  # Short article (500 words)
  python storm_enhanced.py "AI Ethics" --words 500

  # Long article (2000 words, 10 perspectives)
  python storm_enhanced.py "Climate Change" --words 2000 --min-words 1500 --perspectives 10

  # Quick research (3 perspectives, 800 words)
  python storm_enhanced.py "Blockchain" --perspectives 3 --words 800
        """
    )

    parser.add_argument('topic', nargs='?', default='Sustainable Energy',
                       help='Topic to research and write about (default: Sustainable Energy)')
    parser.add_argument('--words', type=int, default=1000,
                       help='Target word count (default: 1000)')
    parser.add_argument('--min-words', type=int, default=500,
                       help='Minimum word count (default: 500)')
    parser.add_argument('--perspectives', type=int, default=5,
                       help='Number of perspectives to explore (default: 5)')
    parser.add_argument('--json-only', action='store_true',
                       help='Output only JSON (no article text)')

    args = parser.parse_args()

    # Create module with configuration
    module = EnhancedStormModule(
        target_words=args.words,
        min_words=args.min_words,
        max_perspectives=args.perspectives
    )

    print(f"\n{'='*80}")
    print(f"Enhanced STORM: {args.topic}")
    print(f"{'='*80}")
    print(f"Configuration: {args.words} words (min: {args.min_words}), {args.perspectives} perspectives")
    print(f"{'='*80}\n")

    results = module.forward(args.topic)

    if args.json_only:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "="*80)
        print("METADATA")
        print("="*80)
        print(f"Word Count: {results['metadata']['word_count']} words (target: {args.words})")
        print(f"Conversations: {results['metadata']['num_conversations']}")
        print(f"Perspectives: {results['metadata']['num_perspectives']}")

        print("\n" + "="*80)
        print("GENERATED ARTICLE")
        print("="*80)
        print(results["article"])

        print("\n" + "="*80)
        print(f"Full results saved to: results.json")
        print("="*80)

        # Save full results to file
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
