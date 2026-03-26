"""
STORM Article Creation State Machine.

This module orchestrates the complete article generation pipeline:
1. Research - Gather Wikipedia links and table of contents
2. Perspectives - Generate multiple viewpoints on the topic
3. Conversation - Create Q&A dialogue from perspectives
4. Outline - Structure the article based on conversations
5. Writing - Generate the final article text

Example:
    >>> machine = ArticleCreationStateMachine("Quantum Computing", lm)
    >>> article = machine.run()
    >>> print(article)
"""
import logging
import os
from typing import Optional, Dict, Any
from research_module import ResearchModule
from perspective_module import PerspectiveModule
from conversation_module import ConversationModule
from outline_creation_module import OutlineCreationModule
from article_writing_module import ArticleWritingModule
import dspy

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

class ArticleCreationStateMachine:
    """
    State machine for orchestrating the article creation pipeline.

    This class manages the complete workflow from research to final article,
    coordinating between different modules and handling state transitions.

    Attributes:
        topic: The main topic for article generation.
        model: The language model instance for DSPy.
        research: Module for conducting research.
        perspective: Module for generating perspectives.
        conversation: Module for Q&A conversations.
        outline_creation: Module for creating article outlines.
        article_writing: Module for writing the final article.
    """

    def __init__(self, topic: str, model: Any):
        """
        Initialize the state machine with a topic and model.

        Args:
            topic: The topic to research and write about.
            model: The DSPy language model instance.
        """
        self.topic = topic
        self.model = model
        self.research = ResearchModule()
        self.perspective = PerspectiveModule()
        self.conversation = ConversationModule()
        self.outline_creation = OutlineCreationModule()
        self.article_writing = ArticleWritingModule()

    def run(self) -> Optional[str]:
        """
        Execute the complete article generation pipeline.

        This method runs all stages of article creation in sequence:
        research, perspective generation, conversations, outline creation,
        and final article writing.

        Returns:
            The generated article text, or None if any stage fails.

        Raises:
            No exceptions are raised; errors are logged and None is returned.
        """
        logging.info(f"Starting the state machine for topic: {self.topic}")

        # Step 1: Conduct Research
        research_result = self.research.forward(self.topic)
        if not research_result:
            logging.error("Research failed or returned no relevant topics.")
            return None

        # Step 2: Generate Perspectives
        perspective_result = self.perspective.forward(self.topic)
        conversation_history = []

        # Step 3: Engage in a Conversation
        for perspective in perspective_result['perspectives']:
            conversation_result = self.conversation.forward(self.topic, perspective, conversation_history)
            conversation_history = conversation_result['conversation_history']

        # Step 4: Create an Article Outline
        outline = self.outline_creation.forward(self.topic, conversation_history)
        if not outline:
            logging.error("Failed to create a draft outline.")
            return None

        # Step 5: Write the Article
        article = self.article_writing.forward(outline, {})
        logging.info(f"Generated article: {article}")
        return article

if __name__ == "__main__":
    topic = "Quantum Computing"
    model = lm  # Updated to use the configured model
    state_machine = ArticleCreationStateMachine(topic, model)
    generated_article = state_machine.run()
    print("Generated article:", generated_article)
