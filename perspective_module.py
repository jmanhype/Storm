import dspy
from dspy import Signature, InputField, OutputField, Module, Predict

# Initialize DSPy settings with a large language model
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key="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")
dspy.settings.configure(lm=claude)

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
