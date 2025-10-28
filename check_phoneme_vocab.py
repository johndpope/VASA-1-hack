#!/usr/bin/env python3
"""
Check the actual vocabulary size of the wav2vec2 phoneme model.
This helps us determine the correct output_dim for the phoneme_head.
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

print("Loading wav2vec2-xlsr-53-espeak-cv-ft model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

print(f"\n{'='*80}")
print("PHONEME MODEL VOCABULARY INFO")
print(f"{'='*80}")

# Get vocab from processor
vocab = processor.tokenizer.get_vocab()
vocab_size = len(vocab)

print(f"\nVocabulary size: {vocab_size}")
print(f"Model config vocab_size: {model.config.vocab_size}")

# Get the actual output dimension
if hasattr(model, 'lm_head'):
    lm_head_out_features = model.lm_head.out_features
    print(f"LM head output features: {lm_head_out_features}")

print(f"\n{'='*80}")
print("VOCABULARY TOKENS")
print(f"{'='*80}")

# Sort vocab by ID
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

print("\nFirst 20 tokens:")
for token, idx in sorted_vocab[:20]:
    print(f"  {idx:3d}: '{token}'")

print("\n...")

print(f"\nLast 20 tokens:")
for token, idx in sorted_vocab[-20:]:
    print(f"  {idx:3d}: '{token}'")

print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")

print(f"""
Update vasa_model.py TalkVidAudioProjection.__init__ line ~154:

BEFORE:
    self.phoneme_head = nn.Linear(dim, 50)  # ❌ Wrong vocab size

AFTER:
    self.phoneme_head = nn.Linear(dim, {vocab_size})  # ✅ Correct vocab size

This matches the wav2vec2 model's actual vocabulary.
""")

print(f"\n{'='*80}")
