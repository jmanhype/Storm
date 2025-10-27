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
        content = " ".join([answer for _, answer in conversation_history])
        full_article = ""
        target_token_length = 800
        min_paragraph_length = 65
        while len(full_article.split()) < target_token_length:
            prediction = self.process_article(topic=topic, content=content, prompt=prompt)
            if hasattr(prediction, 'full_article'):
                generated_text = prediction.full_article.strip()
                paragraphs = generated_text.split('\n')
                for paragraph in paragraphs:
                    if len(paragraph.split()) >= min_paragraph_length:
                        full_article += "\n\n" + paragraph
                    else:
                        prompt += " " + paragraph
                if len(full_article.split()) >= target_token_length:
                    break
            else:
                logging.error("Failed to generate a segment.")
                break
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
    module = ResearchAndConversationModule()
    topic = "Sustainable Energy"
    results = module.forward(topic)
    print("Integrated Research, Conversation, Perspectives, and Article Outputs:")
    print(json.dumps(results, indent=4))
