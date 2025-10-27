import logging
import json
import os
from pydantic import BaseModel
import dspy
from utils import fetch_wikipedia_links, fetch_table_of_contents

logging.basicConfig(level=logging.INFO)

# Get OpenRouter API key from environment
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not found")

# Configuration for the large model with OpenRouter
claude = dspy.OpenAI(
    model="anthropic/claude-3-haiku",
    api_key=openrouter_api_key,
    api_base="https://openrouter.ai/api/v1"
)
dspy.settings.configure(lm=claude)

class LinkData(BaseModel):
    links: list[str]

    def to_json(self):
        return json.dumps(self.links)

class TableOfContents(BaseModel):
    sections: list[str]

    def to_json(self):
        return json.dumps(self.sections)

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

class ResearchModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.research_predict = dspy.ChainOfThought(ResearchSignature)
        self.generate_toc_predict = dspy.ChainOfThought(GenerateTableOfContentsSignature)
        self.perspective_predict = dspy.Predict(PerspectiveSignature)

    def forward(self, topic):
        related_topics = fetch_wikipedia_links(topic)
        table_of_contents = fetch_table_of_contents(topic)
        prediction = self.research_predict(
            topic=topic,
            related_topics=LinkData(links=related_topics).to_json(),
            table_of_contents=TableOfContents(sections=table_of_contents).to_json()
        )
        perspectives = self.perspective_predict(topic=topic)

        results = {
            "topic": topic,
            "related_topics": related_topics,
            "table_of_contents": table_of_contents,
            "perspectives": perspectives.get('perspectives', '').split("\n") if perspectives else []
        }
        
        if prediction and hasattr(prediction, '_completions') and prediction._completions:
            logging.info(f"Raw Prediction: {prediction}")
            logging.info(f"Predictions received: {prediction}")

            # Generate the table of contents using the rationale
            toc_prediction = self.generate_toc_predict(
                topic=topic,
                related_topics=LinkData(links=related_topics).to_json(),
                rationale=prediction.rationale
            )
            if toc_prediction and hasattr(toc_prediction, '_completions') and toc_prediction._completions:
                logging.info(f"Generated Table of Contents: {toc_prediction.table_of_contents}")
                results["table_of_contents"] = toc_prediction.table_of_contents
            else:
                logging.warning("Failed to generate the table of contents.")

        return results

if __name__ == "__main__":
    module = ResearchModule()
    result = module.forward("Quantum Computing")
    if result:
        print("Processing complete. Results:")
        print(json.dumps(result, indent=4))
    else:
        print("Processing failed.")
