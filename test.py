import classla
import time
# classla.download('sl')
nlp = classla.Pipeline('sl', processors='tokenize,pos,lemma')
text = 'France Prešeren je bil rojen v Vrbi. ' * 1000
# Patch to time each processor
for name, proc in nlp.processors.items():
    original_process = proc.process
    def make_timed(n, orig):
        def timed(doc):
            start = time.time()
            result = orig(doc)
            print(f'  {n}: {time.time() - start:.3f}s')
            return result
        return timed
    proc.process = make_timed(name, original_process)
print('First run:')
doc = nlp(text)
print('\nSecond run (should be cached):')
doc = nlp(text + ' Additional text to avoid full cache hits.')
print('\nCache stats:')
for name, proc in nlp.processors.items():
    if hasattr(proc, 'cache_stats'):
        print(f'  {name}: {proc.cache_stats()}')