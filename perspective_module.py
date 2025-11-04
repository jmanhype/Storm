"""
Perspective Generation Module for STORM.

This module generates multiple perspectives on a given topic to enable
comprehensive multi-viewpoint article creation.

Example:
    >>> module = PerspectiveModule()
    >>> result = module.forward("Climate Change")
    >>> print(result['perspectives'])
    ['Scientific perspective', 'Economic perspective', ...]
"""
import os
from typing import Dict, List, Any
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
    """
    Module for generating multiple perspectives on a topic.

    This module uses DSPy to generate diverse viewpoints that can be used
    to create comprehensive, multi-faceted articles.
    """

    def __init__(self):
        """Initialize the perspective generation module."""
        super().__init__()
        self.predict = dspy.Predict(PerspectiveSignature)

    def forward(self, topic: str) -> Dict[str, Any]:
        """
        Generate multiple perspectives for a given topic.

        Args:
            topic: The topic to generate perspectives for.

        Returns:
            A dictionary containing:
                - topic: The input topic
                - perspectives: A list of perspective strings

        Example:
            >>> module = PerspectiveModule()
            >>> result = module.forward("Artificial Intelligence")
            >>> print(len(result['perspectives']))
            5
        """
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
