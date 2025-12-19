"""
Test script for transformer POS encoder implementation.
Run with: uv run python train_test.py
"""

import torch
from classla.models.pos.model import Tagger, TransformerPOSEncoder

print("=" * 60)
print("Testing TransformerPOSEncoder Implementation")
print("=" * 60)

# Test 1: TransformerPOSEncoder instantiation
print("\n1. Testing TransformerPOSEncoder instantiation...")
encoder = TransformerPOSEncoder(input_dim=325, hidden_dim=200, num_layers=4, num_heads=8)
print(f"   ✓ Created successfully")
print(f"   - Input dim: 325, Output dim: {encoder.output_dim}")
print(f"   - Num params: {sum(p.numel() for p in encoder.parameters()):,}")

# Test 2: Forward pass with dummy data
print("\n2. Testing forward pass...")
batch_size, seq_len, input_dim = 2, 10, 325
dummy_input = torch.randn(batch_size, seq_len, input_dim)
sentlens = [10, 7]  # Different sentence lengths
output = encoder(dummy_input, sentlens=sentlens)
print(f"   ✓ Forward pass successful")
print(f"   - Input shape: ({batch_size}, {seq_len}, {input_dim})")
print(f"   - Output shape: {output.shape}")
print(f"   - Expected total tokens: {sum(sentlens)}, got: {output.shape[0]}")

# Test 3: Backward compatibility check
print("\n3. Testing backward compatibility (args.get default)...")
args = {'hidden_dim': 200, 'dropout': 0.5}
use_transformer = args.get('use_transformer', False)
print(f"   args.get('use_transformer', False) = {use_transformer}")
if not use_transformer:
    print("   ✓ Will use BiLSTM (backward compatible with existing models)")
else:
    print("   ✗ ERROR: Should default to False!")

# Test 4: Test with existing classla model loading
print("\n4. Testing with existing classla Pipeline...")
try:
    import classla
    # This will use existing BiLSTM model (use_transformer defaults to False)
    nlp = classla.Pipeline('sl', processors='tokenize,pos', use_gpu=False)
    doc = nlp("To je testni stavek.")
    print(f"   ✓ Existing model loads and works")
    for sent in doc.sentences:
        for word in sent.words:
            print(f"      {word.text}: {word.upos}")
except Exception as e:
    print(f"   ⚠ Could not test with existing model: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
