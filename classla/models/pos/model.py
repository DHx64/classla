import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

from classla.models.common.biaffine import BiaffineScorer
from classla.models.common.hlstm import HighwayLSTM
from classla.models.common.dropout import WordDropout
from classla.models.common.vocab import CompositeVocab
from classla.models.common.char_model import CharacterModel


class TransformerPOSEncoder(nn.Module):
    """
    Transformer encoder for POS tagging.
    Drop-in replacement for Highway BiLSTM that processes sequences in parallel.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=4, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2  # Match BiLSTM bidirectional output
        self.dropout = dropout

        # Normalize input embeddings (critical for transformer stability)
        self.input_norm = nn.LayerNorm(input_dim)

        # Project input to transformer dimension
        self.input_proj = nn.Linear(input_dim, self.output_dim)
        # Initialize projection properly
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.proj_dropout = nn.Dropout(dropout)

        # Positional encoding (sinusoidal - more stable than learned for small data)
        self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(512, self.output_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def _create_sinusoidal_encoding(self, max_len, d_model):
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x, mask=None, sentlens=None):
        """
        Args:
            x: PackedSequence or tensor of shape [batch, seq_len, input_dim]
            mask: Padding mask [batch, seq_len] where True = padding
            sentlens: List of sentence lengths (used if x is PackedSequence)

        Returns:
            Tensor of shape [total_tokens, output_dim] (flattened like BiLSTM output)
        """
        # Handle PackedSequence input (for compatibility with existing code)
        if isinstance(x, PackedSequence):
            # Unpack to padded tensor
            x_padded, lengths = pad_packed_sequence(x, batch_first=True)
            batch_size, max_len, input_dim = x_padded.shape
            # Move lengths to same device as x_padded
            lengths = lengths.to(x_padded.device)

            # Create padding mask from lengths
            mask = torch.arange(max_len, device=x_padded.device).unsqueeze(0) >= lengths.unsqueeze(1)
        else:
            x_padded = x
            batch_size, max_len, _ = x_padded.shape
            if sentlens is not None:
                lengths = torch.tensor(sentlens, device=x_padded.device)
                mask = torch.arange(max_len, device=x_padded.device).unsqueeze(0) >= lengths.unsqueeze(1)

        # Normalize input embeddings (critical!)
        x_normed = self.input_norm(x_padded)

        # Project input
        x_proj = self.input_proj(x_normed)
        x_proj = self.proj_dropout(x_proj)

        # Add positional encoding
        x_proj = x_proj + self.pos_encoding[:, :max_len, :]

        # Apply transformer
        out = self.transformer(x_proj, src_key_padding_mask=mask)
        out = self.output_norm(out)

        # Flatten output to match BiLSTM format [total_tokens, hidden_dim * 2]
        # Only keep non-padded tokens
        if isinstance(x, PackedSequence) or sentlens is not None:
            outputs = []
            for i, length in enumerate(lengths):
                outputs.append(out[i, :length])
            return torch.cat(outputs, dim=0)
        else:
            return out.view(-1, self.output_dim)


class Tagger(nn.Module):
    def __init__(self, args, vocab, emb_matrix=None, share_hid=False):
        super().__init__()

        self.vocab = vocab
        self.args = args
        self.share_hid = share_hid
        self.unsaved_modules = []

        def add_unsaved_module(name, module):
            self.unsaved_modules += [name]
            setattr(self, name, module)

        # input layers
        input_size = 0
        if self.args['word_emb_dim'] > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(len(vocab['word']), self.args['word_emb_dim'], padding_idx=0)
            input_size += self.args['word_emb_dim']

        if not share_hid:
            # upos embeddings
            self.upos_emb = nn.Embedding(len(vocab['upos']), self.args['tag_emb_dim'], padding_idx=0)

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            self.charmodel = CharacterModel(args, vocab)
            self.trans_char = nn.Linear(self.args['char_hidden_dim'], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']

        if self.args['pretrain']:    
            # pretrained embeddings, by default this won't be saved into model file
            add_unsaved_module('pretrained_emb', nn.Embedding.from_pretrained(torch.from_numpy(emb_matrix), freeze=True))
            self.trans_pretrained = nn.Linear(emb_matrix.shape[1], self.args['transformed_dim'], bias=False)
            input_size += self.args['transformed_dim']
        
        # recurrent layers - either BiLSTM or Transformer
        self.use_transformer = self.args.get('use_transformer', False)

        if self.use_transformer:
            self.encoder = TransformerPOSEncoder(
                input_dim=input_size,
                hidden_dim=self.args['hidden_dim'],
                num_layers=self.args.get('transformer_layers', 4),
                num_heads=self.args.get('transformer_heads', 8),
                ff_dim=self.args.get('transformer_ff_dim', 1024),
                dropout=self.args['dropout']
            )
        else:
            self.taggerlstm = HighwayLSTM(input_size, self.args['hidden_dim'], self.args['num_layers'], batch_first=True, bidirectional=True, dropout=self.args['dropout'], rec_dropout=self.args['rec_dropout'], highway_func=torch.tanh)
            self.taggerlstm_h_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))
            self.taggerlstm_c_init = nn.Parameter(torch.zeros(2 * self.args['num_layers'], 1, self.args['hidden_dim']))

        self.drop_replacement = nn.Parameter(torch.randn(input_size) / np.sqrt(input_size))

        # classifiers
        self.upos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'])
        self.upos_clf = nn.Linear(self.args['deep_biaff_hidden_dim'], len(vocab['upos']))
        self.upos_clf.weight.data.zero_()
        self.upos_clf.bias.data.zero_()

        if share_hid:
            clf_constructor = lambda insize, outsize: nn.Linear(insize, outsize)
        else:
            self.xpos_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['deep_biaff_hidden_dim'] if not isinstance(vocab['xpos'], CompositeVocab) else self.args['composite_deep_biaff_hidden_dim'])
            self.ufeats_hid = nn.Linear(self.args['hidden_dim'] * 2, self.args['composite_deep_biaff_hidden_dim'])
            clf_constructor = lambda insize, outsize: BiaffineScorer(insize, self.args['tag_emb_dim'], outsize)

        if isinstance(vocab['xpos'], CompositeVocab):
            self.xpos_clf = nn.ModuleList()
            for l in vocab['xpos'].lens():
                self.xpos_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))
        else:
            self.xpos_clf = clf_constructor(self.args['deep_biaff_hidden_dim'], len(vocab['xpos']))
            if share_hid:
                self.xpos_clf.weight.data.zero_()
                self.xpos_clf.bias.data.zero_()

        self.ufeats_clf = nn.ModuleList()
        for l in vocab['feats'].lens():
            if share_hid:
                self.ufeats_clf.append(clf_constructor(self.args['deep_biaff_hidden_dim'], l))
                self.ufeats_clf[-1].weight.data.zero_()
                self.ufeats_clf[-1].bias.data.zero_()
            else:
                self.ufeats_clf.append(clf_constructor(self.args['composite_deep_biaff_hidden_dim'], l))

        # criterion
        self.crit = nn.CrossEntropyLoss(ignore_index=0) # ignore padding

        self.drop = nn.Dropout(args['dropout'])
        self.worddrop = WordDropout(args['word_dropout'])

    def forward(self, word, word_mask, wordchars, wordchars_mask, upos, xpos, ufeats, pretrained, word_orig_idx, sentlens, wordlens, word_string, postprocessor=None):
        
        def pack(x):
            return pack_padded_sequence(x, sentlens, batch_first=True)
        
        inputs = []
        if self.args['word_emb_dim'] > 0:
            word_emb = self.word_emb(word)
            word_emb = pack(word_emb)
            inputs += [word_emb]

        if self.args['pretrain']:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs += [pretrained_emb]

        def pad(x):
            return pad_packed_sequence(PackedSequence(x, word_emb.batch_sizes), batch_first=True)[0]

        if self.args['char'] and self.args['char_emb_dim'] > 0:
            char_reps = self.charmodel(wordchars, wordchars_mask, word_orig_idx, sentlens, wordlens)
            char_reps = PackedSequence(self.trans_char(self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs += [char_reps]

        lstm_inputs = torch.cat([x.data for x in inputs], 1)
        lstm_inputs = self.worddrop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)
        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        if self.use_transformer:
            # Transformer encoder handles PackedSequence and returns flattened output
            lstm_outputs = self.encoder(lstm_inputs, sentlens=sentlens)
        else:
            lstm_outputs, _ = self.taggerlstm(lstm_inputs, sentlens, hx=(self.taggerlstm_h_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous(), self.taggerlstm_c_init.expand(2 * self.args['num_layers'], word.size(0), self.args['hidden_dim']).contiguous()))
            lstm_outputs = lstm_outputs.data

        upos_hid = F.relu(self.upos_hid(self.drop(lstm_outputs)))
        upos_pred = self.upos_clf(self.drop(upos_hid))

        upos = pack(upos).data
        loss = self.crit(upos_pred.view(-1, upos_pred.size(-1)), upos.view(-1))

        if self.share_hid:
            xpos_hid = upos_hid
            ufeats_hid = upos_hid

            clffunc = lambda clf, hid: clf(self.drop(hid))
        else:
            xpos_hid = F.relu(self.xpos_hid(self.drop(lstm_outputs)))
            ufeats_hid = F.relu(self.ufeats_hid(self.drop(lstm_outputs)))

            if self.training:
                upos_emb = self.upos_emb(upos)
            else:
                upos_emb = self.upos_emb(upos_pred.max(1)[1])

            clffunc = lambda clf, hid: clf(self.drop(hid), self.drop(upos_emb))

        xpos = pack(xpos).data
        if isinstance(self.vocab['xpos'], CompositeVocab):
            xpos_preds = []
            for i in range(len(self.vocab['xpos'])):
                xpos_pred = clffunc(self.xpos_clf[i], xpos_hid)
                loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos[:, i].view(-1))
                xpos_preds.append(pad(xpos_pred).max(2, keepdim=True)[1])
            max_xpos_value = torch.cat(xpos_preds, 2)

        else:
            xpos_pred = clffunc(self.xpos_clf, xpos_hid)
            padded_xpos_pred = pad(xpos_pred)
            if postprocessor is not None:
                max_xpos_value = postprocessor.process_xpos(padded_xpos_pred, word_string)
            else:
                max_xpos_value = padded_xpos_pred.max(2)[1]
            loss += self.crit(xpos_pred.view(-1, xpos_pred.size(-1)), xpos.view(-1))

        padded_upos_pred = pad(upos_pred)
        # if postprocessor.processor and postprocessor.processor.hypothesis_dictionary_upos:
        if postprocessor is not None:
            max_upos_value = postprocessor.process_upos(padded_upos_pred, word_string, max_xpos_value)
            preds = [max_upos_value]
        else:
            preds = [padded_upos_pred.max(2)[1]]

        preds.append(max_xpos_value)
        ufeats_preds = []
        ufeats = pack(ufeats).data
        for i in range(len(self.vocab['feats'])):
            ufeats_pred = clffunc(self.ufeats_clf[i], ufeats_hid)
            loss += self.crit(ufeats_pred.view(-1, ufeats_pred.size(-1)), ufeats[:, i].view(-1))
            ufeats_preds.append(pad(ufeats_pred).max(2, keepdim=True)[1])

        # if postprocessor.processor and postprocessor.processor.hypothesis_dictionary_feats:
        if postprocessor is not None:
            preds.append(postprocessor.process_feats(ufeats_preds, word_string, max_xpos_value, max_upos_value))
        else:
            preds.append(torch.cat(ufeats_preds, 2))

        return loss, preds
