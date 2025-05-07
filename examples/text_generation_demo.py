#!/usr/bin/env python
"""
Demonstration of text generation capabilities using BridgeNLP.

This example shows how to use the new text generation adapters 
(summarization, paraphrasing, and translation) in BridgeNLP.
"""

import spacy
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

try:
    from bridgenlp.pipeline import Pipeline
    from bridgenlp.config import BridgeConfig
    from bridgenlp.adapters.hf_summarization import HuggingFaceSummarizationBridge
    from bridgenlp.adapters.hf_paraphrase import HuggingFaceParaphraseBridge
    from bridgenlp.adapters.hf_translation import HuggingFaceTranslationBridge
except ImportError:
    raise ImportError(
        "BridgeNLP not installed or missing dependencies. "
        "Install with: pip install -e '.[huggingface]'"
    )


def main():
    """Run the text generation demo."""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]BridgeNLP Text Generation Demo[/bold cyan]\n"
        "Showcasing summarization, paraphrasing, and translation capabilities",
        border_style="cyan"
    ))
    
    # Sample text for demonstration
    long_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, 
    as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals.
    
    The term "artificial intelligence" had previously been used to describe 
    machines that mimic and display "human" cognitive skills that are associated 
    with the human mind, such as "learning" and "problem-solving". This definition 
    has since been rejected by major AI researchers who now describe AI in terms 
    of rationality and acting rationally, which does not limit how intelligence 
    can be articulated.
    
    AI applications include advanced web search engines, recommendation systems, 
    understanding human speech, self-driving cars, automated decision-making and 
    competing at the highest level in strategic game systems. As machines become 
    increasingly capable, tasks considered to require "intelligence" are often 
    removed from the definition of AI, a phenomenon known as the AI effect.
    """
    
    # Initialize spaCy
    console.print("\n[bold]Initializing spaCy...[/bold]")
    nlp = spacy.load("en_core_web_sm")
    
    # Step 1: Text Summarization
    console.print("\n[bold]Step 1: Text Summarization[/bold]")
    console.print("Creating summarization adapter...")
    
    try:
        # Create a summarization adapter
        summarizer = HuggingFaceSummarizationBridge(
            model_name="facebook/bart-large-cnn",
            max_length=75,
            min_length=30
        )
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Tip: Install transformers with:[/yellow] pip install transformers torch")
        return
    
    # Process text with summarization
    console.print("Generating summary...")
    try:
        summary_result = summarizer.from_text(long_text)
    except Exception as e:
        console.print(f"[bold red]Error generating summary:[/bold red] {str(e)}")
        summary_result = BridgeResult(tokens=["No summary available"], roles=[{"role": "SUMMARY", "text": "Summary generation failed"}])
    
    # Extract and display the summary
    summary_text = summary_result.roles[0]["text"] if summary_result.roles else "No summary generated."
    
    console.print(Panel(
        f"[bold]Original Text:[/bold] ({len(long_text.split())} words)\n" +
        long_text + "\n\n" +
        f"[bold]Summary:[/bold] ({len(summary_text.split())} words)\n" +
        summary_text,
        title="Summarization Result",
        border_style="green"
    ))
    
    # Step 2: Paraphrasing
    console.print("\n[bold]Step 2: Paraphrasing[/bold]")
    console.print("Creating paraphrasing adapter...")
    
    try:
        # Create a paraphrasing adapter
        paraphraser = HuggingFaceParaphraseBridge(
            model_name="tuner007/pegasus_paraphrase",
            num_return_sequences=3,
            temperature=0.7
        )
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Tip: Install transformers with:[/yellow] pip install transformers torch")
        return
    
    # Extract first sentence for paraphrasing
    first_sentence = "Artificial intelligence is intelligence demonstrated by machines, as opposed to natural intelligence displayed by humans."
    
    # Process text with paraphrasing
    console.print("Generating paraphrases...")
    try:
        paraphrase_result = paraphraser.from_text(first_sentence)
    except Exception as e:
        console.print(f"[bold red]Error generating paraphrases:[/bold red] {str(e)}")
        paraphrase_result = BridgeResult(tokens=["No paraphrase available"], roles=[{"role": "PARAPHRASE", "text": "Paraphrase generation failed", "variant": 1}])
    
    # Extract and display the paraphrases
    paraphrases = [role["text"] for role in paraphrase_result.roles] if paraphrase_result.roles else ["No paraphrases generated."]
    
    console.print(Panel(
        f"[bold]Original Sentence:[/bold]\n" +
        first_sentence + "\n\n" +
        "[bold]Paraphrases:[/bold]\n" +
        "\n".join([f"{i+1}. {p}" for i, p in enumerate(paraphrases)]),
        title="Paraphrasing Result",
        border_style="blue"
    ))
    
    # Step 3: Translation
    console.print("\n[bold]Step 3: Translation[/bold]")
    console.print("Creating translation adapter...")
    
    try:
        # Create a translation adapter
        translator = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-en-fr"
        )
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Tip: Install transformers with:[/yellow] pip install transformers torch")
        return
    
    # Short text for translation
    simple_text = "Artificial intelligence is transforming the way we live and work."
    
    # Process text with translation
    console.print("Generating translations...")
    try:
        translation_result = translator.from_text(simple_text)
    except Exception as e:
        console.print(f"[bold red]Error generating translation:[/bold red] {str(e)}")
        translation_result = BridgeResult(tokens=["No translation available"], roles=[{
            "role": "TRANSLATION", 
            "text": "Translation failed",
            "source_lang": "en",
            "target_lang": "fr",
            "original_text": simple_text
        }])
    
    # Extract and display the translation
    translation_text = translation_result.roles[0]["text"] if translation_result.roles else "No translation generated."
    
    console.print(Panel(
        f"[bold]English Text:[/bold]\n" +
        simple_text + "\n\n" +
        f"[bold]French Translation:[/bold]\n" +
        translation_text,
        title="Translation Result",
        border_style="yellow"
    ))
    
    # Step 4: Combining in a Pipeline
    console.print("\n[bold]Step 4: Pipeline Integration[/bold]")
    console.print("Creating a pipeline with summarization and translation...")
    
    # Create a pipeline config
    config = BridgeConfig(cache_results=True, collect_metrics=True)
    
    # Create a pipeline with summarization followed by translation
    try:
        # Creating a new translator for English to Spanish
        es_translator = HuggingFaceTranslationBridge(
            model_name="Helsinki-NLP/opus-mt-en-es"
        )
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("[yellow]Tip: Install transformers with:[/yellow] pip install transformers torch")
        return
    
    # Create and run the pipeline
    pipeline = Pipeline([summarizer, es_translator], config)
    
    console.print("Running the pipeline (summarize -> translate)...")
    try:
        pipeline_result = pipeline.from_text(long_text)
    except Exception as e:
        console.print(f"[bold red]Error running pipeline:[/bold red] {str(e)}")
        pipeline_result = BridgeResult(tokens=["Pipeline failed"], roles=[{
            "role": "TRANSLATION", 
            "text": "Pipeline processing failed",
            "source_lang": "en",
            "target_lang": "es"
        }])
    
    # Extract the final translated summary
    translated_summary = pipeline_result.roles[0]["text"] if pipeline_result.roles else "No result generated."
    
    console.print(Panel(
        f"[bold]Original Long Text (English):[/bold] ({len(long_text.split())} words)\n" +
        "Text has been summarized and translated to Spanish:\n\n" +
        f"[bold]Translated Summary (Spanish):[/bold]\n" +
        translated_summary,
        title="Pipeline Result",
        border_style="magenta"
    ))
    
    # Display performance metrics
    console.print("\n[bold]Performance Metrics:[/bold]")
    metrics = pipeline.get_metrics()
    
    for key, value in metrics.items():
        if isinstance(value, float):
            console.print(f"  - {key}: {value:.4f}")
        else:
            console.print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()