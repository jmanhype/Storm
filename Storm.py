import dspy
import requests
import re
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import spacy
from sentence_transformers import SentenceTransformer, util
import textstat
from dspy.teleprompt import BootstrapFewShot
import logging
import json
import random
# Configure logging at the beginning of your script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Evaluation Functions
def readability_and_coherence(generated_answer):
    readability_score = textstat.flesch_reading_ease(generated_answer)
    return readability_score

model = SentenceTransformer('all-MiniLM-L6-v2')

def relevancy_score(generated_answer, expected_answer):
    generated_embedding = model.encode(generated_answer, convert_to_tensor=True)
    expected_embedding = model.encode(expected_answer, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(generated_embedding, expected_embedding)
    return similarity.item()

def comprehensiveness(generated_answer, key_concepts):
    score = sum(1 for concept in key_concepts if concept in generated_answer) / len(key_concepts)
    return score

def combined_metric(generated_answer, expected_answer, key_concepts):
    relevancy = relevancy_score(generated_answer, expected_answer)
    comprehensiveness_score = comprehensiveness(generated_answer, key_concepts)
    readability = readability_and_coherence(generated_answer)
    final_score = 0.5 * relevancy + 0.3 * comprehensiveness_score + 0.2 * readability
    return final_score

# Initialize OpenAI LLM client
openai_llm = dspy.OpenAI(model='gpt-3.5-turbo', max_tokens=300, api_key='your-api-key-here')
dspy.settings.configure(lm=openai_llm)


# Load the dataset from the file
with open('my_dataset.json', 'r') as file:
    data = json.load(file)

# Print the first few queries to check their content
for item in data[:5]:
    print(f"Query: {item['question']}")


# Convert each entry into a DSPy example
dataset = [dspy.Example(inputs={'query': item['question']},
                        outputs={"expected_answer": item['answer']},
                        metadata=item.get('metadata', {})).with_inputs('query') for item in data]

# Split your dataset into train and dev sets as needed
trainset, devset = dataset[:int(len(dataset) * 0.8)], dataset[int(len(dataset) * 0.8):]

print(f"Trainset size: {len(trainset)}, Devset size: {len(devset)}")


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Define Signatures
class DiscoverySignature(dspy.Signature):
    query = dspy.InputField()
    titles = dspy.OutputField()

class ScrapingSignature(dspy.Signature):
    titles = dspy.InputField()
    articles = dspy.OutputField()

class PreprocessingSignature(dspy.Signature):
    articles = dspy.InputField()
    preprocessed_articles = dspy.OutputField()

class ParsingSignature(dspy.Signature):
    preprocessed_articles = dspy.InputField()
    parsed_data = dspy.OutputField()

class SummarizerSignature(dspy.Signature):
    article_content = dspy.InputField()
    summary = dspy.OutputField()

# Define Modules
class OpenAISummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summarize = dspy.Predict(SummarizerSignature)

    def forward(self, article_content):
        summarized = self.summarize(article_content=article_content)
        summary_text = summarized.summary
        return {'summary': summary_text}

class DataDiscoveryAgent(dspy.Module):
    def forward(self, query):
        wikipedia_search_url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
        response = requests.get(wikipedia_search_url, params=params)
        data = response.json() if response.status_code == 200 else {}
        titles = [result["title"] for result in data.get("query", {}).get("search", [])]
        return {'titles': titles}


class DataScrapingAgent(dspy.Module):
    def forward(self, titles):
        articles = {}
        for title in titles:
            formatted_title = title.replace(" ", "_")
            wikipedia_summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_title}"
            response = requests.get(wikipedia_summary_url)
            data = response.json() if response.status_code == 200 else {}
            articles[title] = data.get("extract", "")
        return {'articles': articles}

class DataPreprocessingAgent(dspy.Module):
    def forward(self, articles):
        preprocessed_articles = {}
        for title, content in articles.items():
            cleaned_content = re.sub('<[^<]+?>', '', content)
            cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
            preprocessed_articles[title] = cleaned_content
        return {'preprocessed_articles': preprocessed_articles}

class QuestionAnsweringAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "deepset/roberta-base-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
        self.qa_pipeline = pipeline("question-answering", model=self.model, tokenizer=self.tokenizer)

    def forward(self, preprocessed_articles):
        parsed_data = {}
        questions = ["What is the main topic?", "Who is involved?", "Where did it take place?"]
        for title, content in preprocessed_articles.items():
            answers = {}
            for question in questions:
                result = self.qa_pipeline(question=question, context=content)
                answers[question] = result['answer']
            parsed_data[title] = answers
        return {'parsed_data': parsed_data}

# Workflow Manager

class WorkflowManager(dspy.Module):
    def __init__(self):
        super().__init__()
        self.discovery_agent = DataDiscoveryAgent()
        self.scraping_agent = DataScrapingAgent()
        self.preprocessing_agent = DataPreprocessingAgent()
        self.summarizer = OpenAISummarizer()
        self.question_answering_agent = QuestionAnsweringAgent()
        self.dataset_path = 'my_dataset.json'  # Path to the dataset file
        self.logger = logging.getLogger(__name__)

    def load_fallback_query(self):
        """Load a random query from the dataset as a fallback."""
        try:
            with open(self.dataset_path, 'r') as file:
                data = json.load(file)
                if data:
                    # Select a random entry's question for fallback
                    random_entry = random.choice(data)
                    fallback_query = random_entry.get('question', '')
                    self.logger.info(f"Loaded fallback query: {fallback_query}")
                    return fallback_query
        except Exception as e:
            self.logger.error(f"Failed to load fallback query from dataset: {e}")
        return ''  # Return an empty string if unable to load a fallback query

    def forward(self, **inputs):
        # Extract 'query' from the inputs dictionary; use the fallback mechanism if not found
        query = inputs.get('query', None)
        if query is None or query == 'Default Query if Missing':
            query = self.load_fallback_query()  # Load a fallback query from the dataset
            if not query:  # If no fallback query could be loaded, log a warning
                self.logger.warning("No fallback query loaded. Proceeding with 'Default Query if Missing'.")
                query = 'Default Query if Missing'

        self.logger.info(f"Received query in WorkflowManager: {query}")  # Log the received or fallback query

        titles = self.discovery_agent(query=query)['titles']
        self.logger.info(f"Titles discovered: {titles}")

        articles = self.scraping_agent(titles=titles)['articles']
        self.logger.info(f"Articles scraped: {articles.keys()}")

        preprocessed_articles = self.preprocessing_agent(articles=articles)['preprocessed_articles']
        self.logger.info("Articles preprocessed.")

        summaries = {title: self.summarizer(article_content=content)['summary'] for title, content in preprocessed_articles.items()}
        self.logger.info("Summaries generated.")

        parsed_data = self.question_answering_agent(preprocessed_articles=summaries)['parsed_data']
        self.logger.info(f"Final parsed data: {parsed_data.keys()}")

        return parsed_data


# Stateful Workflow Manager
class StatefulWorkflowManager(WorkflowManager):
    def __init__(self):
        super().__init__()
        self.context = {'history': [], 'topics': set(), 'unresolved': []}

    def forward(self, query):
        result = super().forward(query)
        self.update_context(query, result)
        return result

    def update_context(self, query, result):
        self.context['history'].append((query, result))
        for title, details in result.items():
            for question, answer in details.items():
                doc = nlp(answer)
                for ent in doc.ents:
                    self.context['topics'].add(ent.text)
                if self.is_topic_unresolved(answer):
                    unresolved_topic = self.extract_unresolved_topic(answer)
                    self.context['unresolved'].append(unresolved_topic)

    def is_topic_unresolved(self, text):
        indicators_of_uncertainty = ["it is unclear", "unknown", "uncertain", "undetermined", "more research is needed", "remains to be seen", "debated", "controversial"]
        return any(phrase in text.lower() for phrase in indicators_of_uncertainty) or "?" in text

    def extract_unresolved_topic(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            return ent.text
        for chunk in doc.noun_chunks:
            return chunk.text
        return "specific topic"

# Recursive Ensemble Manager
class RecursiveEnsembleManager(dspy.Module):
    def __init__(self, workflows, reduce_fn):
        super().__init__()
        self.workflows = workflows
        self.reduce_fn = reduce_fn

    def forward(self, query, depth=0, max_depth=3):
        results = [workflow(query) for workflow in self.workflows]
        combined_result = self.reduce_fn(results)
        if self.should_follow_up(combined_result, depth):
            follow_up_query = self.generate_follow_up_query(combined_result)
            return self.forward(follow_up_query, depth+1, max_depth)
        else:
            return combined_result

    def should_follow_up(self, combined_result, depth):
        return depth < 3 and any(self.is_topic_unresolved(answer) for details in combined_result.values() for answer in details.values())

    def generate_follow_up_query(self, combined_result):
        topic_counts = {}
        for details in combined_result.values():
            for answer in details.values():
                if self.is_topic_unresolved(answer):
                    unresolved_topic = self.extract_unresolved_topic(answer)
                    topic_counts[unresolved_topic] = topic_counts.get(unresolved_topic, 0) + 1
        follow_up_topic = max(topic_counts, key=topic_counts.get) if topic_counts else None
        return f"Can you provide more details about {follow_up_topic}?" if follow_up_topic else "Could you elaborate more on this?"

    def is_topic_unresolved(self, text):
        indicators_of_uncertainty = [
            "it is unclear", "unknown", "uncertain", "undetermined",
            "more research is needed", "remains to be seen", "debated", "controversial"
        ]
        return any(phrase in text.lower() for phrase in indicators_of_uncertainty) or "?" in text

    def extract_unresolved_topic(self, text):
        doc = nlp(text)
        for ent in doc.ents:
            return ent.text
        for chunk in doc.noun_chunks:
            return chunk.text
        return "specific topic"


# Combine Results Function
def merge_results(results):
    merged_result = {}
    for result in results:
        for title, details in result.items():
            if title not in merged_result:
                merged_result[title] = details
            else:
                for question, answer in details.items():
                    existing_answers = set(merged_result[title][question].split("; "))
                    new_answers = set(answer.split("; "))
                    all_unique_answers = list(existing_answers.union(new_answers))
                    merged_result[title][question] = "; ".join(all_unique_answers)
    return merged_result

# After defining all modules and classes...

# Define the evaluation metric function
def evaluation_metric(example, model_output, *args, **kwargs):
    # Debug statement to print the input example
    print(f"Evaluating example with input: {example.inputs}")

    # Extract the generated answer from the model's output
    generated_answer = model_output.get('answer', '')
    # Extract the expected answer from the example
    expected_answer = example.outputs['expected_answer']
    # Extract key concepts from the example's metadata
    key_concepts = example.metadata.get('key_concepts', [])

    # Calculate the combined metric score based on the generated and expected answers, and key concepts
    score = combined_metric(generated_answer, expected_answer, key_concepts)

    # Debug statement to print the generated answer, expected answer, and the calculated score
    print(f"Generated answer: {generated_answer}, Expected answer: {expected_answer}, Score: {score}")

    # Return the calculated score
    return score





# Define examples for training and evaluation
# Define examples for training and evaluation using dspy.Example and .with_inputs()




# Add more examples as needed

# Initialize the teleprompter with the evaluation metric and examples
teleprompter = BootstrapFewShot(
    metric=evaluation_metric,
    max_labeled_demos=5,
    max_bootstrapped_demos=10
)


# Compile your system with the teleprompter
compiled_system = teleprompter.compile(student=WorkflowManager(), trainset=trainset)

# Use the compiled system to process queries
query = "What are the benefits of solar energy?"
final_result = compiled_system(inputs={"query": query})
print("Final Result:", final_result)
