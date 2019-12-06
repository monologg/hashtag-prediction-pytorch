import torch
import torch.nn as nn
from transformers import AlbertModel
from transformers.modeling_albert import AlbertPreTrainedModel


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.relu(x)
        return self.linear(x)


class HashtagClassifier(AlbertPreTrainedModel):
    def __init__(self, bert_config, args):
        super(HashtagClassifier, self).__init__(bert_config)
        self.albert = AlbertModel.from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.num_labels = bert_config.num_labels

        self.text_classifier = FCLayer(bert_config.hidden_size, 100, args.dropout_rate, use_activation=False)
        self.img_classifier = FCLayer(512*7*7, 100, args.dropout_rate, use_activation=False)

        self.label_classifier = FCLayer(200, self.num_labels, args.dropout_rate, use_activation=True)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, img_features):
        outputs = self.albert(input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        pooled_output = outputs[1]  # [CLS]
        text_tensors = self.text_classifier(pooled_output)
        
        # NOTE Concat text feature and img features [512, 7, 7]
        img_flatten = torch.flatten(img_features, start_dim=1)
        img_tensors = self.img_classifier(img_flatten)

        # Concat -> fc_layer
        logits = self.label_classifier(torch.cat((text_tensors, img_tensors), -1))
        logits = torch.sigmoid(logits)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits, labels.float())

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
