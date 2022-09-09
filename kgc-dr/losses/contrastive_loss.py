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

class SeqContrastiveLoss():
    def __call__(self, query_emb, passage_emb, in_batch_mask, neg_passage_emb = None, T=1):
        """
        query_emb: [L1, D]
        passage_emb: [L1, D]
        in_batch_mask: [L1, L1]
        neg_passage_emb: [nxL1, D]
        """
        assert query_emb.dim() == passage_emb.dim() and query_emb.size(0) == passage_emb.size(0)
        dtype = in_batch_mask.dtype 
        device = in_batch_mask.device

        targets = torch.arange(0, query_emb.size(0), dtype=torch.int64, device=device)
        if neg_passage_emb != None:
            neg_passage_mask = torch.ones(query_emb.size(0), neg_passage_emb.size(0)).to(device=device, dtype=dtype)
            logit_mask = torch.cat([in_batch_mask, neg_passage_mask], dim=1)
            all_passage_emb = torch.cat([passage_emb, neg_passage_emb], dim=0)
        else:
            logit_mask = in_batch_mask
            all_passage_emb = passage_emb
        print()
        # change the 0-1 logits_mask to -\infity-0 logit_mask
        logit_mask = (1.0 - logit_mask) * torch.finfo(dtype).min
        logits = (query_emb @ all_passage_emb.t()) + logit_mask
        logits /= T
        
        #print("logits: ", logits[0][:logits.size(1)//2])
        #print("logits: ", logits[0][logits.size(1)//2:])
        #print(logits.size())
        
        return F.cross_entropy(logits, targets)


    

if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False)
    import sys 
    sys.path += ["../"]
    bz, D = 128, 786
    qs = torch.randn(bz, D)
    ps_1 = qs + torch.normal(mean=torch.zeros_like(qs), std=.01)
    ps_2 = qs + torch.normal(mean=torch.zeros_like(qs), std=1.)
    ps = []
    for x1, x2 in zip(ps_1, ps_2):
        ps.append(x1.unsqueeze(0))
        ps.append(x2.unsqueeze(0))
    ps = torch.cat(ps, dim=0)
    print(ContrastiveLoss()(qs, ps))
    #print(ContrastiveLoss()(qs, ps_2))
    """
    torch.set_printoptions(sci_mode=False)
    import sys 
    sys.path += ["../"]
    from dataset.user_sequential_dataset import UserSequentialDataset

    print("hi")
    for bz in [4, 16, 64, 128, 256]:
        L = 4
        L1 = bz*L
        D = 786
        n = 1
        seq_lengths = [L]*bz
        assert L1 == sum(seq_lengths)
        in_batch_mask = torch.ones(L1, L1)
        masked_indices = UserSequentialDataset.get_in_batch_masked_indices(seq_lengths)
        in_batch_mask[masked_indices] = 0. 

        query_emb = torch.randn(L1, D)
        passage_emb = query_emb + torch.normal(mean=torch.zeros_like(query_emb), std=.01)
        neg_passage_emb = query_emb + torch.normal(mean=torch.zeros_like(query_emb), std=1.0)

        loss = SeqContrastiveLoss()(query_emb, passage_emb, in_batch_mask, neg_passage_emb, T=10)
        print(f"bz: {bz}, loss: {loss.item():.3f}")
        
        #break
    """