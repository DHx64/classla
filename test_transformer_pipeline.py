"""
Test the full classla pipeline with transformer POS tagger.
Run on RunPod: python test_transformer_pipeline.py
"""
import time
import shutil
import os

# Step 1: Copy transformer model to classla resources location
print("=" * 60)
print("Setting up transformer model for pipeline test")
print("=" * 60)

transformer_model = "/workspace/classla/saved_models/pos/sl_transformer"
target_dir = "/root/classla_resources/sl/pos/"
target_model = os.path.join(target_dir, "standard_transformer.pt")

# Backup original and copy transformer
original_model = os.path.join(target_dir, "standard.pt")
backup_model = os.path.join(target_dir, "standard_bilstm_backup.pt")

if os.path.exists(transformer_model):
    print(f"Found transformer model at: {transformer_model}")

    # Backup original BiLSTM model if not already backed up
    if os.path.exists(original_model) and not os.path.exists(backup_model):
        print(f"Backing up original BiLSTM model to: {backup_model}")
        shutil.copy(original_model, backup_model)

    # Copy transformer model as standard.pt (so pipeline uses it)
    print(f"Copying transformer model to: {original_model}")
    shutil.copy(transformer_model, original_model)
    print("Done!")
else:
    print(f"ERROR: Transformer model not found at {transformer_model}")
    print("Make sure training has saved a model.")
    exit(1)

# Step 2: Test the pipeline
print("\n" + "=" * 60)
print("Testing Pipeline with Transformer POS Tagger")
print("=" * 60)

import classla

# Create pipeline - it will now load the transformer model
print("\nLoading pipeline (this will load the transformer model)...")
start_load = time.time()
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma', use_gpu=True)
print(f"Pipeline loaded in {time.time() - start_load:.2f}s")

# Test texts
test_texts = [
    "France Prešeren je bil slovenski pesnik.",
    "Ljubljana je glavno mesto Slovenije.",
    "Danes je lep sončen dan.",
    "Slovenija je majhna država v srednji Evropi.",
    "Triglav je najvišja gora v Sloveniji.",
]

# Warm up run
print("\nWarm-up run...")
_ = nlp(test_texts[0])

# Single sentence test
print("\n--- Single Sentence Test ---")
text = test_texts[0]
start = time.time()
doc = nlp(text)
elapsed = time.time() - start
print(f"Text: '{text}'")
print(f"Time: {elapsed*1000:.1f}ms")
print("Results:")
for sent in doc.sentences:
    for word in sent.words:
        print(f"  {word.text:15} UPOS={word.upos:6} XPOS={word.xpos:15} Lemma={word.lemma}")

# Batch test - 100 sentences
print("\n--- Batch Test (100 sentences) ---")
batch_text = " ".join(test_texts * 20)  # 100 sentences
start = time.time()
doc = nlp(batch_text)
elapsed = time.time() - start
num_sents = len(doc.sentences)
num_words = sum(len(s.words) for s in doc.sentences)
print(f"Processed {num_sents} sentences, {num_words} words in {elapsed:.3f}s")
print(f"Speed: {num_sents/elapsed:.1f} sentences/sec, {num_words/elapsed:.1f} words/sec")

# Large batch test - 1000 sentences
print("\n--- Large Batch Test (1000 sentences) ---")
large_text = " ".join(test_texts * 200)  # 1000 sentences
start = time.time()
doc = nlp(large_text)
elapsed = time.time() - start
num_sents = len(doc.sentences)
num_words = sum(len(s.words) for s in doc.sentences)
print(f"Processed {num_sents} sentences, {num_words} words in {elapsed:.3f}s")
print(f"Speed: {num_sents/elapsed:.1f} sentences/sec, {num_words/elapsed:.1f} words/sec")

# Per-processor timing
print("\n--- Per-Processor Timing (1000 sentences) ---")
for name, proc in nlp.processors.items():
    original_process = proc.process
    def make_timed(n, orig):
        def timed(doc):
            start = time.time()
            result = orig(doc)
            print(f"  {n}: {time.time() - start:.3f}s")
            return result
        return timed
    proc.process = make_timed(name, original_process)

doc = nlp(large_text)

# Step 3: Compare with BiLSTM (optional - restore backup)
print("\n" + "=" * 60)
print("Comparison Summary")
print("=" * 60)
print("\nTo compare with BiLSTM, run:")
print(f"  cp {backup_model} {original_model}")
print("  python test_transformer_pipeline.py")
print("\nTo restore transformer:")
print(f"  cp {transformer_model} {original_model}")
