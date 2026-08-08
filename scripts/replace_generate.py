import re

p = "models/acap.py"
src = open(p).read()

old = '''    def _generate(
        self,
        concept_embeds: torch.Tensor,
        roi_features: torch.Tensor,
        batch_size: int,
        max_length: int,
    ) -> List[str]:'''

new = '''    def _generate(
        self,
        concept_embeds: torch.Tensor,
        roi_features: torch.Tensor,
        batch_size: int,
        max_length: int,
        num_iter: int = 10,
    ) -> List[str]:'''

# Find the _generate method and replace everything from def _generate to the next def or end
start = src.find("    def _generate(\n")
if start == -1:
    print("ERROR: _generate not found")
    exit(1)

# Find the end: the next line starting with 4 spaces + "def " at same indent
end = src.find("\n    def ", start + 10)
if end == -1:
    end = len(src)

old_block = src[start:end]

new_block = '''    def _generate(
        self,
        concept_embeds: torch.Tensor,
        roi_features: torch.Tensor,
        batch_size: int,
        max_length: int,
        num_iter: int = 10,
    ) -> List[str]:
        # Iterative Mask-Predict decoding (Ghazvininejad et al., 2019).
        # The decoder is a bidirectional MLM, not autoregressive. Single-pass
        # all-mask argmax is crude (every position independent, no refinement).
        # Instead: predict all, keep the most confident, re-mask the rest, repeat.
        # Over num_iter iterations the caption is progressively refined, giving
        # the model more context each iteration (like the partial masking in
        # training with mlm_mask_prob < 1.0).
        L = self.word_seq_length
        mask_id = self.vinvl.mask_token_id

        tokens = torch.full((batch_size, L), mask_id, dtype=torch.long,
                            device=self.device)

        for t in range(num_iter):
            word_embeds = self.vinvl.embeddings(input_ids=tokens)
            seq_out, _ = self.vinvl(
                word_embeddings=word_embeds,
                concept_embeddings=concept_embeds,
                roi_features=roi_features,
            )
            logits = self.vinvl.lm_head(seq_out[:, :L, :])
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

            n_unmask = int(L * (t + 1) / num_iter)

            if t < num_iter - 1:
                for b in range(batch_size):
                    conf = confidence[b].clone()
                    already = tokens[b] != mask_id
                    conf[already] = float('inf')
                    n_keep_masked = L - n_unmask
                    if n_keep_masked > 0:
                        _, low_conf_idx = conf.sort()
                        keep_mask = torch.zeros(L, dtype=torch.bool, device=self.device)
                        keep_mask[low_conf_idx[:n_keep_masked]] = True
                        fill = ~keep_mask
                        tokens[b, fill] = predicted[b, fill]
                    else:
                        tokens[b] = predicted[b]
            else:
                tokens = predicted

        captions = []
        for ids in tokens:
            caption = self.vinvl.tokenizer.decode(ids, skip_special_tokens=True)
            captions.append(caption)

        return captions
'''

src = src[:start] + new_block + src[end:]
open(p, "w").write(src)
print("REPLACED _generate with iterative mask-predict")
