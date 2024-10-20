import torch

tag2id = {'O': 0, 'B-MOUNTAIN': 1, 'I-MOUNTAIN': 2}
id2tag = {v: k for k, v in tag2id.items()}


def predict(model, tokenizer, sentence):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(
        sentence,
        return_tensors='pt',
        truncation=True,
        padding=True
    ).to(device)

    # get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)

    # convert predictions to tags
    predicted_tags = []
    tokens_from_ids = []
    word_ids = inputs.word_ids()

    input_ids = inputs['input_ids'][0].tolist()

    for idx, pred in enumerate(predictions[0]):
        if word_ids[idx] is not None:  # skip special tokens
            predicted_tags.append(id2tag[pred.item()])
            tokens_from_ids.append(tokenizer.decode([input_ids[idx]]))

    return predicted_tags, tokens_from_ids


def join_subwords(tokens):
    result = []
    for token in tokens:
        if token.startswith("##"):
            # append the subword without "##" to the last word in the result list
            if len(result):
                result[-1] += token[2:]
            else:
                result.append(token)
        else:
            # Append a new word to the result list
            result.append(token)
    return ' '.join(result)


def extract_mountains(tokens, tags):
    mountains = []
    current_mountain = []

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-MOUNTAIN'):
            if current_mountain:
                mountains.append(join_subwords(current_mountain))
            current_mountain = [token]
        elif tag.startswith('I-MOUNTAIN'):
            current_mountain.append(token)
        else:
            if current_mountain:
                mountains.append(join_subwords(current_mountain))
                current_mountain = []

    if current_mountain:
        mountains.append(join_subwords(current_mountain))

    return join_subwords(mountains)


def infer(model, tokenizer, sentence):
    model.eval()
    predicted_tags, tokens_from_ids = predict(model, tokenizer, sentence)
    mountains = extract_mountains(tokens_from_ids, predicted_tags)
    return tokens_from_ids, predicted_tags, mountains
