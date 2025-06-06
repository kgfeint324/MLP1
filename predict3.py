
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import EmotionClassifier as EmotionClassifier60, inv_label_mapping as inv_label_mapping60, label_mapping
from model2 import EmotionClassifier as EmotionClassifier6, fine_to_coarse
import ipywidgets as widgets
from IPython.display import display, clear_output

#대분류 감정
coarse_map = {k.upper(): v.upper() for k, v in fine_to_coarse.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
project_path = "/content/drive/MyDrive/emotion_project"

emotion_names_60 = {
    "E10": "분노", "E11": "툴툴대는", "E12": "좌절한", "E13": "짜증내는", "E14": "방어적인",
    "E15": "악의적인", "E16": "안달하는", "E17": "구역질 나는", "E18": "노여워하는", "E19": "성가신",
    "E20": "슬픔", "E21": "실망한", "E22": "비통한", "E23": "후회되는", "E24": "우울한",
    "E25": "마비된", "E26": "염세적인", "E27": "눈물이 나는", "E28": "낙담한", "E29": "환멸을 느끼는",
    "E30": "불안", "E31": "두려운", "E32": "스트레스 받는", "E33": "취약한", "E34": "혼란스러운",
    "E35": "당혹스러운", "E36": "회의적인", "E37": "걱정스러운", "E38": "조심스러운", "E39": "초조한",
    "E40": "상처", "E41": "질투하는", "E42": "배신당한", "E43": "고립된", "E44": "충격 받은",
    "E45": "불우한", "E46": "희생된", "E47": "억울한", "E48": "괴로워하는", "E49": "버려진",
    "E50": "당황", "E51": "고립된(당황한)", "E52": "남의 시선을 의식하는", "E53": "외로운", "E54": "열등감",
    "E55": "죄책감", "E56": "부끄러운", "E57": "혐오스러운", "E58": "한심한", "E59": "혼란스러운(당황한)",
    "E60": "기쁨", "E61": "감사하는", "E62": "신뢰하는", "E63": "편안한", "E64": "만족스러운",
    "E65": "흥분", "E66": "느긋", "E67": "안도", "E68": "신이 난", "E69": "자신하는"
}

emotion_names_6 = {
    "A01": "분노", "A02": "슬픔", "A03": "불안",
    "A04": "상처", "A05": "당황", "A06": "기쁨"
}

inv_label_mapping_6 = {
    0: "A01", 1: "A02", 2: "A03", 3: "A04",
    4: "A05", 5: "A06"
}

model60 = EmotionClassifier60("klue/bert-base").to(device)
model60.load_state_dict(torch.load(f"{project_path}/best_model0.pt", map_location=device))
model60.eval()

model6 = EmotionClassifier6("klue/bert-base").to(device)
model6.load_state_dict(torch.load(f"{project_path}/best_model.pt", map_location=device))
model6.eval()

def predict3(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)

    with torch.no_grad():
        probs60 = F.softmax(model60(inputs["input_ids"], inputs["attention_mask"]), dim=1).squeeze()
        probs6 = F.softmax(model6(inputs["input_ids"], inputs["attention_mask"]), dim=1).squeeze()

    top3_60 = torch.topk(probs60, k=3)
    top3_6 = torch.topk(probs6[:6], k=3) 

    result = []
    for i in range(3):
        idx = top3_60.indices[i].item()
        code = inv_label_mapping60[idx]
        name = emotion_names_60.get(code, "Unknown")
        score = top3_60.values[i].item()
        group = coarse_map.get(code, "")
        if group in [inv_label_mapping_6[i.item()] for i in top3_6.indices]:
            result.append((group, code, name, score))

    print(f"📝 입력 문장: {sentence}")
    if result:
        print("🎯 필터링 된 감정:")
        for group, fine, name, score in sorted(result, key=lambda x: -x[3]):
            group_name = emotion_names_6.get(group, "")
            print(f"{group}({group_name}) - {fine}({name}) → 확률: {score:.4f}")
    else:
        print("⚠️ 대분류와 일치하는 감정 없음 (소분류 Top3):")
        for i in range(3):
            idx = top3_60.indices[i].item()
            code = inv_label_mapping60[idx]
            name = emotion_names_60.get(code, "Unknown")
            score = top3_60.values[i].item()
            print(f"{code} ({name}) - 확률: {score:.4f}")

text_input = widgets.Text(description="문장:")
output_area = widgets.Output()

def on_submit(text_widget):
    sentence = text_widget.value.strip()
    if sentence:
        with output_area:
            clear_output()
            predict3(sentence)
        text_input.value = ""

text_input.on_submit(on_submit)
display(text_input, output_area)
