import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class ContrastiveLoss():
    def __call__(self, queries, passages):
        """
        queries: [bz, dim]
        passages: [bz * nway, dim]
        """
        assert len(passages) % len(queries) == 0, (passages.shape, queries.shape)
        assert passages.dim() == queries.dim() == 2
        
        passages_per_query = len(passages) // len(queries)
        
        logits =  queries @ passages.t() #[bz, bz*nway]
        targets = torch.arange(0, len(passages), passages_per_query, dtype=torch.int64, device=queries.device)
        
        return F.cross_entropy(logits, targets)
    

if __name__ == "__main__":
    qs = torch.randn(4, 3)
    ps_1 = torch.randn(8, 3)
    ps_2 = torch.randn(12,3)
    print(ContrastiveLoss()(qs, ps_1))
    print(ContrastiveLoss()(qs, ps_2))