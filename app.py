import streamlit as st
import dspy
from dspy import Predict, Signature, InputField, OutputField, Program, Example
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.teleprompt.ensemble import Ensemble  # Correct import for Ensemble
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define stylistic similarity metric
def stylistic_similarity(reference_texts, generated_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([generated_text] + reference_texts)
    similarity_matrix = cosine_similarity(vectors[0:1], vectors[1:])
    return np.mean(similarity_matrix)

def combined_metric(example, pred, trace=None):
    generated_article = pred.generated_content
    readability_score = textstat.flesch_reading_ease(generated_article)
    style_score = stylistic_similarity(st.session_state.reference_articles, generated_article)
    combined_score = 0.5 * readability_score + 0.5 * style_score
    return combined_score

class Assess(dspy.Signature):
    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField()

class TopicSpecificModelSignature(Signature):
    prompt = InputField()
    style_features = InputField()
    generated_content = OutputField()

class TopicSpecificModel(Program):
    def __init__(self, max_tokens=1024, update_word_count=50):
        super().__init__()
        self.max_tokens = max_tokens
        self.update_word_count = update_word_count
        self.generate_article = Predict(TopicSpecificModelSignature, max_tokens=self.max_tokens)

    def forward(self, **inputs):
        prompt = inputs.get('prompt')
        style_features = inputs.get('style_features')
        full_article = ""
        while len(full_article.split()) < 1000:
            generated_segment = self.generate_article(prompt=prompt, style_features=style_features)
            segment_text = generated_segment['generated_content']
            full_article += segment_text + ' '
            prompt = ' '.join(full_article.split()[-self.update_word_count:])
        return full_article

def aggregate_contents(topic_contents):
    aggregated_article = ""
    for idx, content in enumerate(topic_contents):
        if idx == 0:
            aggregated_article += f"Introduction to the topic:\n{content}\n"
        else:
            aggregated_article += f"\nFurther Insights:\n{content}\n"
    aggregated_article += "\nConclusion: This article covered various topics, providing insights and information on each."
    return aggregated_article

def select_best_content(contents):
    best_score = -1
    best_content = None

    for content in contents:
        readability_score = textstat.flesch_reading_ease(content)
        style_score = stylistic_similarity(st.session_state.reference_articles, content)

        # Placeholder for AI feedback, implement your logic here
        correct_score = engaging_score = 0.5  # Dummy scores, replace with actual AI feedback logic

        combined_score = 0.25 * readability_score + 0.25 * style_score + 0.25 * correct_score + 0.25 * engaging_score

        if combined_score > best_score:
            best_score = combined_score
            best_content = content

    return best_content

# Streamlit UI
st.title('DSPy Content Generator')

api_key = st.text_input('Enter your OpenAI API Key:', type='password')

if api_key:
    dspy.settings.configure(lm=dspy.OpenAI(model='gpt-3.5-turbo', api_key=api_key))

    st.session_state.reference_articles = st.text_area('Enter Reference Articles (separated by new lines):').split('\n')

    prompts = st.text_area('Enter Prompts (separated by new lines):').split('\n')

    if st.button('Generate Content'):
        with st.spinner('Generating content...'):
            teleprompter = BootstrapFewShotWithRandomSearch(
                metric=combined_metric,
                teacher_settings={'lm': dspy.settings.lm},
                max_bootstrapped_demos=10,
                max_labeled_demos=5,
                max_rounds=1,
                num_candidate_programs=16,
                num_threads=6
            )

            compiled_models = []
            for prompt in prompts:
                model = TopicSpecificModel()
                compiled_model = teleprompter.compile(student=model, trainset=[Example(prompt=prompt, generated_content="")])
                compiled_models.append(compiled_model)

            ensemble_teleprompter = Ensemble(reduce_fn=select_best_content)
            ensembled_program = ensemble_teleprompter.compile(compiled_models)

            ensembled_contents = [ensembled_program(prompt=prompt) for prompt in prompts]

            final_ensembled_article = aggregate_contents(ensembled_contents)

            st.subheader('Generated Article')
            st.write(final_ensembled_article)
