import gbnet 
import pandas as pd 
import pickle 

rp = gbnet.aux.Reporter()

rp.report("Init") 
file_paths = [f"gbnet_{v}.csv" for v in ["ents", "rels", "evid"]] 
dfs = [pd.read_csv(fp) for fp in file_paths] 
 
model = gbnet.models.ORNORModel(*dfs, nchains=3)
rp.report("Model loaded")

# rp.report("Init sampling single core")
# model.sample(N=500, thin=1, njobs=1)
# rp.report("done sampling single core")

rp.report("Init sampling multicore")
model.sample(N=500, thin=1, njobs=3)
rp.report("done sampling multicore")
