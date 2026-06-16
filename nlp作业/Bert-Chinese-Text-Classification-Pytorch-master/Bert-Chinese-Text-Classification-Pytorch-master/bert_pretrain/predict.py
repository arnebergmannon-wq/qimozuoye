import torch
from importlib import import_module


dataset = 'THUCNews'
model_name = 'bert'

x = import_module('models.' + model_name)
config = x.Config(dataset)

model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=config.device))
model.eval()


def predict(text):
    token = config.tokenizer.tokenize(text)
    token = ['[CLS]'] + token
    seq_len = len(token)

    token_ids = config.tokenizer.convert_tokens_to_ids(token)
    pad_size = config.pad_size

    if len(token_ids) < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - len(token_ids))
        token_ids += [0] * (pad_size - len(token_ids))
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    input_ids = torch.LongTensor([token_ids]).to(config.device)
    seq_len = torch.LongTensor([seq_len]).to(config.device)
    mask = torch.LongTensor([mask]).to(config.device)

    with torch.no_grad():
        outputs = model((input_ids, seq_len, mask))
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return config.class_list[pred], confidence


if __name__ == '__main__':
    examples = [
        "房间很干净，服务态度也很好，下次还会来。",
        "房间隔音太差，晚上根本睡不着。",
        "酒店离地铁站很近，出行非常方便。",
        "前台服务态度很差，房间还有异味。",
        "位置不错，但是房间太旧了。"
    ]

    for text in examples:
        label, confidence = predict(text)
        print("输入：", text)
        print("预测结果：", label)
        print("置信度：{:.4f}".format(confidence))
        print("-" * 50)