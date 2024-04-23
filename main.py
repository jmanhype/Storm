import logging
from research_module import ResearchModule
from perspective_module import PerspectiveModule
from conversation_module import ConversationModule
from outline_creation_module import OutlineCreationModule
from article_writing_module import ArticleWritingModule
import dspy

# Initialize DSPy settings with a large language model
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="")
dspy.settings.configure(lm=claude)

class ArticleCreationStateMachine:
    def __init__(self, topic, model):
        self.topic = topic
        self.model = model
        self.research = ResearchModule()
        self.perspective = PerspectiveModule()
        self.conversation = ConversationModule()
        self.outline_creation = OutlineCreationModule()
        self.article_writing = ArticleWritingModule()

    def run(self):
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
    model = claude  # Updated to use the configured model
    state_machine = ArticleCreationStateMachine(topic, model)
    generated_article = state_machine.run()
    print("Generated article:", generated_article)
