#!/usr/bin/env python
"""
Demonstration of question answering using BridgeNLP.

This example shows how to use the HuggingFaceQABridge adapter
to perform question answering and integrate it with spaCy.
"""

import spacy
from spacy import displacy

try:
    from bridgenlp.adapters.hf_qa import HuggingFaceQABridge
except ImportError:
    raise ImportError(
        "Hugging Face dependencies not found. Install with: "
        "pip install bridgenlp[huggingface]"
    )


def main():
    """Run the question answering demo."""
    # Create a spaCy pipeline
    nlp = spacy.load("en_core_web_sm")
    
    # Create a question answering bridge
    qa_bridge = HuggingFaceQABridge()
    
    # Define a context and questions
    context = """
    The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
    It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
    Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially
    criticized by some of France's leading artists and intellectuals for its design, but it
    has become a global cultural icon of France and one of the most recognizable structures
    in the world. The Eiffel Tower is the most-visited paid monument in the world.
    """
    
    questions = [
        "Who designed the Eiffel Tower?",
        "When was the Eiffel Tower built?",
        "Why was the Eiffel Tower built?",
        "How was the Eiffel Tower initially received?"
    ]
    
    # Process the context
    doc = nlp(context)
    
    # Answer each question
    for question in questions:
        print(f"\nQuestion: {question}")
        
        # Set the question and process the document
        qa_bridge.set_question(question)
        processed_doc = qa_bridge.from_spacy(doc)
        
        # Print the answer
        if processed_doc._.nlp_bridge_roles:
            answer = processed_doc._.nlp_bridge_roles[0]
            print(f"Answer: {answer['text']} (confidence: {answer['score']:.2f})")
            
            # Highlight the answer in the document
            if 'start_token' in answer and 'end_token' in answer:
                answer_span = doc[answer['start_token']:answer['end_token']]
                print(f"Context: ...{doc[max(0, answer_span.start - 10):min(len(doc), answer_span.end + 10)].text}...")
        else:
            print("No answer found.")


if __name__ == "__main__":
    main()
