from transformers import LlamaForCausalLM, LlamaTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Modell und Tokenizer laden
model_name = "meta-llama/Llama-3"  # Beispiel-Modellname
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = LlamaTokenizer.from_pretrained(model_name)

# Bereitgestellte Trainings- und Testdaten laden
data_folder = "Training/data"
train_data = f"{data_folder}/train_data.json"
test_data = f"{data_folder}/test_data.json"

# Datensätze laden
train_dataset = load_dataset('json', data_files=train_data)['train']
test_dataset = load_dataset('json', data_files=test_data)['train']

# Tokenisierung vorbereiten
def tokenize_function(examples):
    return tokenizer(examples['query'], truncation=True, padding="max_length", max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Trainingsargumente definieren
training_args = TrainingArguments(
    output_dir="./results",          # Ausgabeordner für das Modell
    evaluation_strategy="epoch",    # Modell wird nach jeder Epoche evaluiert
    learning_rate=5e-5,             # Lernrate
    per_device_train_batch_size=8,  # Batch-Größe pro Gerät
    num_train_epochs=3,             # Anzahl der Epochen
    weight_decay=0.01,              # Gewichtungsabfall
    logging_dir="./logs",           # Logging-Ordner
    save_strategy="epoch"           # Speichern nach jeder Epoche
)

# Trainer definieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test
)

# Fine-Tuning starten
trainer.train()

trainer.save_model("./fine_tuned_llama")
print("Fine-Tuning abgeschlossen und Modell gespeichert!")