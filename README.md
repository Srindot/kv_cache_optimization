# 


📋 Summary Table
Level	Notebook	Core Strategy
Token	attention_sink	Keep initial + recent tokens
Token	minicache	Merge cache across layers
Token	H20	Evict low-attention tokens
Token	PyramidKV	Progressive layer reduction
Token	window_sliding_cache	Fixed sliding window
Model	Attention Grouping and Sharing	Share KV heads (GQA)
Model	Architecture Alteration	Compress prompt cache
Model	non_transformer	Replace transformer (RWKV/Mamba)
System	quantization	Quantize cache to INT8/INT4
System	smooth_quant	Advanced quantization
System	lorc	Low-rank + quantization
System	scheduling	Batch scheduling optimization
System	memory	Engine-level optimization
