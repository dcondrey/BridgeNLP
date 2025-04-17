#!/bin/bash
# Comprehensive test script for BridgeNLP

# Set up environment
echo "Setting up test environment..."
export PYTHONPATH=.

# Run basic tests
echo "Running basic tests..."
pytest -xvs tests/test_bridge_result.py tests/test_aligner.py tests/test_spacy_pipe.py

# Run stress tests
echo "Running stress tests..."
pytest -xvs tests/test_stress.py

# Run adapter tests if dependencies are available
echo "Running adapter tests (if dependencies are available)..."
pytest -xvs tests/test_spacy_ner.py
pytest -xvs tests/test_allen_coref.py
pytest -xvs tests/test_hf_srl.py
pytest -xvs tests/test_nltk_adapter.py

# Run coverage tests
echo "Running coverage tests..."
pytest --cov=bridgenlp tests/

# Print coverage report
echo "Coverage report:"
pytest --cov=bridgenlp --cov-report=term-missing tests/
