'''
    最初的叠加模型：
class BertCNNEmb(nn.Module):
    """
        自己的模型：基于BERT-CNN-Embedding
    """

    def __init__(self, args):
        super(BertCNNEmb, self).__init__()
        bert_path = args.bert_path
        bert_input_size = args.bert_input_size
        class_num = args.class_num
        filter_sizes = args.filter_sizes
        filter_num = args.filter_num
        dropout = args.dropout
        rel_embedding_dim = args.rel_embedding_dim  # 未定义

        self.class_num = class_num
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (size, bert_input_size)) for size in filter_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_layer = nn.Linear(filter_num * len(filter_sizes), class_num)

        self.dropout1 = nn.Dropout(dropout)
        self.cls_layer_emb = nn.Linear(rel_embedding_dim, class_num, bias=False)

    def forward(self, x, x_emb):
        tokens = self.tokenizer(x, padding=True)
        input_ids = torch.tensor(tokens["input_ids"])
        attention_mask = torch.tensor(tokens["attention_mask"])
        with torch.no_grad():
            last_hidden_states = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state, bert_output = last_hidden_states['last_hidden_state'], last_hidden_states[
            'pooler_output']
        x = last_hidden_state.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x_logits = self.cls_layer(x)

        x_emb = self.dropout1(x_emb)
        x_emb_logits = self.cls_layer_emb(x_emb)
        logits = torch.add(x_logits, x_emb_logits)/2.0
        return logits


'''