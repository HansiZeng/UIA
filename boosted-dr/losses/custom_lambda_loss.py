import torch
import torch.nn as nn 

class CustomLambdaLoss(nn.Module):
    def __init__(self, apply_boosted_weight, distill_teacher_score, boosted_k=.5, boosted_b=.5, reduction="mean"):
        super().__init__()
        self.apply_boosted_weight = apply_boosted_weight
        self.distill_teacher_score = distill_teacher_score
        self.reduction = reduction
        self.boosted_k = boosted_k
        self.boosted_b = boosted_b

    def forward(self, y_pred, y_true, boosted_y = None):
        device = y_pred.device
        clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

        # for y_true
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        true_sorted_by_preds.clamp_(min=0.) # what if y_pred < 0 ?

        # for lambda weights
        if self.distill_teacher_score:
            y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
            y_true_sorted.clamp_(min=0.)

            # Here we find the gains, discounts and ideal DCGs per slate.
            pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
            D = torch.log2(1. + pos_idxs.float())[None, :]

            maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D), dim=-1).clamp(min=1e-4)
            G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

            weights = ndcgLoss1_scheme(G, D)  # type: ignore
        else:
            inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)
            weights = torch.abs(inv_pos_idxs.view(1,-1,1) - inv_pos_idxs.view(1,1,-1)) # [1, topk, topk]

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        losses = torch.log(1. + torch.exp(-scores_diffs)) * weights #[bz, topk, topk]

         # for boosted weights 
        if self.apply_boosted_weight:
            boosted_y_sorted_by_preds = torch.gather(boosted_y, dim=1, index=indices_pred)
            boosted_diffs = (boosted_y_sorted_by_preds[:, :, None] - boosted_y_sorted_by_preds[:, None, :]).clamp(min=-clamp_val, max=clamp_val)
            boosted_diffs.masked_fill(torch.isnan(boosted_diffs), 0.)
            boosted_weights = self.boosted_k * torch.sigmoid(-boosted_diffs) + self.boosted_b

            losses *= boosted_weights

        if self.reduction == "sum":
            loss = torch.sum(losses[padded_pairs_mask])
        elif self.reduction == "mean":
            loss = torch.mean(losses[padded_pairs_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

def ndcgLoss1_scheme(G, D, *args):
    return (G / D)[:, :, None]

if __name__ == "__main__":
    y_pred = torch.tensor([[106., 110., 109., 108.],
                [110., 106., 109., 108.]])
    y_true = torch.tensor([[1., .5, .33, .25],
                [1., .5, .33, .25]])
    boosted_y = torch.tensor([[103., 111., 110., 108.],
                   [111., 107., 105., 108.]])

    lambda_loss_1 = CustomLambdaLoss(False)
    #print(lambda_loss_1(y_pred, y_true))

    lambda_loss_1 = CustomLambdaLoss(True)
    print(lambda_loss_1(y_pred, y_true, boosted_y))

    new_indices = torch.LongTensor([[1,0,3,2], [3,2,1,0]])
    y_pred = torch.gather(y_pred, dim=-1, index=new_indices)
    y_true = torch.gather(y_true, dim=-1, index=new_indices)
    boosted_y = torch.gather(boosted_y, dim=-1, index=new_indices)
    print(lambda_loss_1(y_pred, y_true, boosted_y))