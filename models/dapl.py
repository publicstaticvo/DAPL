
import copy
from models.gnn import *
from models.bert import BertModel, BertPreTrainedModel


class DAPL(BertPreTrainedModel):
    def __init__(self, config):
        super(DAPL, self).__init__(config)
        self.config = config
        self.model = config.model
        self.bert = BertModel(config)
        if config.model == "linear":
            self.linear = torch.nn.Linear(config.hidden_size * 2, config.feature_size * 2)
        elif config.model in ["gcn", "gat", "agcn"]:
            # self.activate = torch.nn.LeakyReLU(config.leakyrelu)
            self.dependency_embedding = torch.nn.Embedding(config.num_dep, config.hidden_size, padding_idx=0)
            if config.model == "gcn":
                graph_network = GraphConvolution(config.hidden_size, config.hidden_size)
                self.f = torch.nn.ModuleList([copy.deepcopy(graph_network) for _ in range(config.num_layers)])
            elif config.model == "gat":
                graph_network = GraphAttention(config.hidden_size, config.hidden_size, config.leakyrelu)
                self.f = torch.nn.ModuleList([copy.deepcopy(graph_network) for _ in range(config.num_layers)])
            elif config.model == "agcn":
                graph_network = AttentionGNN(config.hidden_size, config.hidden_size)
                self.f = torch.nn.ModuleList([copy.deepcopy(graph_network) for _ in range(config.num_layers)])
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def valid_filter(self, output, valid):
        return torch.max(output.unsqueeze(1).repeat(1, output.shape[-2], 1, 1).masked_fill(valid.unsqueeze(-1).bool(),
                                                                                           float("-inf")), dim=2)[0]

    def extract_entity(self, sequence, e_mask):
        entity_output = sequence.masked_fill(mask=e_mask.unsqueeze(-1).bool(), value=float("-inf"))
        entity_output = torch.max(entity_output, -2)[0]
        return entity_output.type_as(sequence)

    def forward(self, input_ids, e1_mask, e2_mask, attention_mask, valid=None, token_type_ids=None,
                dep_matrix=None, dep_pos=None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                dep_pos=dep_pos)
        # get relation representation
        sequence_output, pooler_output = bert_output[:2]  # batch_size * max_len * hidden_size
        sequence_output = self.dropout(sequence_output)
        if self.model in ["gcn", "gat", "agcn"]:
            sequence_output = self.valid_filter(sequence_output, valid)
            dep_adj_matrix = torch.clamp(dep_matrix, 0, 1)
            for i, layer in enumerate(self.f):
                sequence_output = layer(sequence_output, dep_adj_matrix)
                if self.model == "agcn":
                    sequence_output = sequence_output[0]
                # if i != self.config.num_layers - 1:
                #     sequence_output = self.activate(sequence_output)
                sequence_output = self.dropout(sequence_output)
        e1_h = self.extract_entity(sequence_output, e1_mask)
        e2_h = self.extract_entity(sequence_output, e2_mask)
        relation_embedding = torch.cat((e1_h, e2_h), 1)
        if self.model == "linear":
            relation_embedding = self.linear(relation_embedding)
        return relation_embedding
