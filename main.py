import numpy as np
from collections import defaultdict
 
class LSH:
    def __init__(self, d, n_planes=10, n_tables=5):
        self.d=d; self.L=n_tables; self.K=n_planes
        self.planes=[np.random.randn(n_planes,d) for _ in range(n_tables)]
        self.tables=[defaultdict(list) for _ in range(n_tables)]
        self.data=[]
    def _hash(self, x, table_idx):
        return tuple((self.planes[table_idx]@x>0).astype(int))
    def add(self, x, idx):
        self.data.append(x)
        for i in range(self.L): self.tables[i][self._hash(x,i)].append(idx)
    def query(self, q, top_k=5):
        candidates=set()
        for i in range(self.L): candidates|=set(self.tables[i].get(self._hash(q,i),[]))
        if not candidates: return []
        dists=[(idx,np.linalg.norm(q-self.data[idx])) for idx in candidates]
        return [idx for idx,_ in sorted(dists,key=lambda x:x[1])[:top_k]]
 
class MIPS_LSH:
    """Maximum Inner Product Search via asymmetric LSH."""
    def __init__(self, d, m=3, n_planes=8):
        self.U_max=1.0; self.m=m
        self.lsh=LSH(d+2*m, n_planes)
    def transform_db(self, x):
        norm=np.linalg.norm(x); norms=[norm**2**i for i in range(1,self.m+1)]
        return np.concatenate([x/self.U_max, norms, [0]*self.m])
    def transform_query(self, q):
        return np.concatenate([q/self.U_max, [0]*self.m, [0.5]*self.m])
    def add(self, x, idx): self.lsh.add(self.transform_db(x), idx)
    def search(self, q, top_k=5): return self.lsh.query(self.transform_query(q), top_k)
 
np.random.seed(42); n,d=1000,128
db=np.random.randn(n,d); q=np.random.randn(d)
lsh=LSH(d,n_planes=12,n_tables=8)
for i,x in enumerate(db): lsh.add(x,i)
lsh_results=lsh.query(q,5)
exact=np.argsort([np.linalg.norm(q-x) for x in db])[:5].tolist()
recall=len(set(lsh_results)&set(exact))/5
print(f"LSH results:   {lsh_results}")
print(f"Exact results: {exact}")
print(f"Recall@5: {recall:.2f}")
