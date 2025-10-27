import os
import dspy
from dspy import Signature, InputField, OutputField, Module, Predict

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

class PerspectiveSignature(dspy.Signature):
    topic = dspy.InputField(desc="The main topic for which perspectives are needed")
    perspectives = dspy.OutputField(desc="Generated list of perspectives")

class PerspectiveModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PerspectiveSignature)

    def forward(self, topic):
        # Ensuring that the `topic` is correctly packaged in the call
        response = self.predict(topic=topic)  # The topic is now explicitly passed
        
        # Assuming the model outputs newline-separated perspectives
        if response and 'perspectives' in response:
            perspectives = response['perspectives'].split("\n")
        else:
            perspectives = []
        
        return {
            "topic": topic,
            "perspectives": perspectives
        }

if __name__ == "__main__":
    # Example usage
    perspective_module = PerspectiveModule()
    result = perspective_module.forward("Environmental Sustainability")
    print(result)
