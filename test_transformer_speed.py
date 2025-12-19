import time
import classla
from classla.models.pos.trainer import Trainer
from classla.models.common.pretrain import Pretrain

# Test text
texts = [
    "France Prešeren je bil slovenski pesnik.",
    "Ljubljana je glavno mesto Slovenije.",
    "Danes je lep sončen dan.",
]

# Load transformer model manually for testing
pretrain = Pretrain("~/classla_resources/sl/pretrain/standard.pt")
trainer = Trainer(
    args={'use_transformer': True, 'transformer_layers': 4, 'transformer_heads': 8},
    pretrain=pretrain,
    model_file="~/classla_resources/sl/pos/sl_transformer.pt",
    use_cuda=False
)

# Time it
start = time.time()
# ... run inference
elapsed = time.time() - start
print(f"Processed {len(texts)} sentences in {elapsed:.2f}s")