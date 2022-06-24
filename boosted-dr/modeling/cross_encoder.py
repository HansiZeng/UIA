import os 

from transformers import AutoModelForSequenceClassification, AutoConfig
import torch
import torch.nn.functional as F 
import torch.nn as nn 

class CrossEncoder(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args 
        self.model_name_or_path = self.model_args.model_name_or_path

        self.config = AutoConfig.from_pretrained(self.model_name_or_path)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, config=self.config)

    def forward(self, inputs):
        if inputs["input_ids"].dim() == 2:
            preds = self.classifier(**inputs, return_dict=True)
            scores = preds.logits 
            assert scores.shape[1] == 1
            return scores.view(-1) #[bz]
        elif inputs["input_ids"].dim() == 3:
            bz, nway, seq_len = inputs["input_ids"].size()
            inputs = {k:v.view(bz*nway,seq_len) for k, v in inputs.items()}
            preds = self.classifier(**inputs, return_dict=True)
            scores = preds.logits.view(bz, nway) 
            return scores #[bz, nway]
        else:
            raise ValueError("inputs should have dim either 2 or 3")

    @classmethod
    def from_pretrained(cls, model_path, model_args=None):
        if os.path.isdir(model_path):
            ckpt_path = os.path.join(model_path, "model.pt")
            model_args_path = os.path.join(model_path, "model_args.pt")
            model_args = torch.load(model_args_path)
        elif os.path.isfile(model_path):
            assert model_args != None 
            ckpt_path = model_path
        else:
            raise ValueError("model_path: {} is not expected".format(model_path))
        
        model = cls(model_args)
        print("load pretrained model from local path {}".format(model_path))

        model_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(model_dict)

        return model 

    def save_pretrained(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        model_to_save = self
        torch.save(model_to_save.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(self.model_args, os.path.join(save_dir, "model_args.pt"))


if __name__ == "__main__":
    import sys 
    sys.path += ["./"]
    from arguments import CEModelArguments
    from transformers import HfArgumentParser

    parser = HfArgumentParser((CEModelArguments))
    model_args = parser.parse_args_into_dataclasses()[0]

    model = CrossEncoder(model_args)

    print("config: ", model.config)
    for name, param in model.named_parameters():
        print(name, param.shape)




