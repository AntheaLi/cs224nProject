import torch
import sys

p = sys.argv[1]

a = torch.load('save/'+p+'/checkpoint/DistillBert_DANN')
b = {}
for k in a:
	if 'distilbert' in k:
		b[k[11:]] = a[k]

c = {}

#for k in b:
#	if 'distilbert' in k:
#		c[k[11:]] = b[k]
#	else:
#		c[k] = b[k]

torch.save(b, 'save/'+p+'/checkpoint/pytorch_model.bin')
