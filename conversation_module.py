import logging
import json
from pydantic import BaseModel
import dspy
from utils import fetch_wikipedia_links, fetch_table_of_contents

logging.basicConfig(level=logging.INFO)

claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="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")
dspy.settings.configure(lm=claude)

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

class ResearchAndConversationModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research_module = dspy.ChainOfThought(ResearchSignature)
        self.generate_toc_module = dspy.ChainOfThought(GenerateTableOfContentsSignature)
        self.conversation_module = dspy.ChainOfThought(ConversationSignature)
        self.perspective_predict = dspy.Predict(PerspectiveSignature)

    def forward(self, topic):
        related_topics = fetch_wikipedia_links(topic)
        # Generate Table of Contents
        toc_data = self.generate_toc_module(
            topic=topic,
            related_topics=LinkData(links=related_topics).to_json(),
            rationale="Generate detailed TOC based on key subtopics"
        )
        table_of_contents = toc_data.table_of_contents if toc_data else "No TOC generated"

        perspectives_output = self.perspective_predict(topic=topic)

        conversation_history = [("Initial query", f"Introduction to {topic}")]
        formatted_history = ' '.join([f"{q}: {a}" for q, a in conversation_history])

        conversation_output = self.conversation_module(
            topic=topic,
            perspective=perspectives_output.get('perspectives', ''),
            conversation_history=formatted_history
        )

        updated_history = conversation_history + [(conversation_output.question, conversation_output.answer)]

        return {
            "research": {
                "related_topics": related_topics,
                "table_of_contents": table_of_contents
            },
            "conversation": {
                "next_question": conversation_output.question,
                "answer": conversation_output.answer,
                "history": updated_history
            },
            "perspectives": perspectives_output.perspectives.split("\n") if 'perspectives' in perspectives_output else []
        }

if __name__ == "__main__":
    module = ResearchAndConversationModule()
    topic = "Sustainable Energy"
    results = module.forward(topic)
    print("Integrated Research, Conversation, and Perspectives Outputs:")
    print(json.dumps(results, indent=4))
