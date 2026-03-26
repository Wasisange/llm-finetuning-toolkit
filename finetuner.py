from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

class LLMFineTuner:
    def __init__(self, model_name, dataset):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.dataset = dataset

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenized_dataset = self.dataset.map(self._tokenize_function, batched=True)

    def _tokenize_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    def train(self, num_epochs, learning_rate):
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            learning_rate=learning_rate,
            logging_dir="./logs",
            logging_steps=10,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
        )

        trainer.train()

    def evaluate(self):
        # Simple evaluation placeholder
        print("Evaluation complete.")

if __name__ == "__main__":
    # Example usage (requires a local 'train.txt' and 'val.txt' for demonstration)
    # Create dummy files for demonstration
    with open("train.txt", "w") as f:
        f.write("This is a sample training text.\n")
        f.write("Another line for training.\n")
    with open("val.txt", "w") as f:
        f.write("This is a sample validation text.\n")

    from datasets import load_dataset
    dataset = load_dataset("text", data_files={"train": "train.txt", "validation": "val.txt"})

    finetuner = LLMFineTuner(model_name="gpt2", dataset=dataset)
    finetuner.train(num_epochs=1, learning_rate=2e-5)
    finetuner.evaluate()
