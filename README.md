# LLM Fine-tuning Toolkit

A toolkit for efficient fine-tuning of large language models on custom datasets.

## Features
- Supports various LLM architectures (e.g., Llama, GPT-2, T5).
- Efficient data loading and preprocessing for fine-tuning.
- Integrates with popular deep learning frameworks (PyTorch, Hugging Face Transformers).
- Provides scripts for hyperparameter optimization and evaluation.

## Installation

```bash
git clone https://github.com/Wasisange/llm-finetuning-toolkit.git
cd llm-finetuning-toolkit
pip install -r requirements.txt
```

## Usage

```python
from finetuner import LLMFineTuner
from datasets import load_dataset

# Load a custom dataset
dataset = load_dataset("text", data_files={"train": "train.txt", "validation": "val.txt"})

# Initialize and run the fine-tuner
finetuner = LLMFineTuner(model_name="gpt2", dataset=dataset)
finetuner.train(num_epochs=3, learning_rate=2e-5)
finetuner.evaluate()
```

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
