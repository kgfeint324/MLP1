# Import necessary libraries
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model_sub import EmotionClassifier as EmotionClassifier_sub, inv_label_mapping as inv_label_mapping60
from model_major import EmotionClassifier as EmotionClassifier_major, fine_to_coarse
import ipywidgets as widgets
from IPython.display import display, clear_output

# Mapping fine-grained emotion labels (e.g., E10) to coarse-grained labels (e.g., A01), converted to uppercase
coarse_map = {k.upper(): v.upper() for k, v in fine_to_coarse.items()}

# Set device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# Path to saved models
project_path = "/content/drive/MyDrive/emotion_project"

# Dictionary mapping fine-grained emotion codes (E##) to Korean emotion names
emotion_names_sub = {
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

# Dictionary mapping coarse-grained emotion codes (A##) to Korean emotion names
emotion_names_major = {
    "A01": "분노", "A02": "슬픔", "A03": "불안",
    "A04": "상처", "A05": "당황", "A06": "기쁨"
}

emotion_emojis = {
    "A01": "😠",  # 분노
    "A02": "😢",  # 슬픔
    "A03": "😨",  # 불안
    "A04": "💔",  # 상처
    "A05": "😳",  # 당황
    "A06": "😊",  # 기쁨
}

# Reverse mapping for coarse-grained emotion classification (index to label)
inv_label_mapping_6 = {
    0: "A01", 1: "A02", 2: "A03", 3: "A04",
    4: "A05", 5: "A06"
}

# Load fine-grained emotion classifier model (sub)
model_sub = EmotionClassifier_sub("klue/bert-base").to(device)
model_sub.load_state_dict(torch.load(f"{project_path}/best_model_sub.pt", map_location=device))
model_sub.eval()

# Load coarse-grained emotion classifier model (major)
model_major = EmotionClassifier_major("klue/bert-base").to(device)
model_major.load_state_dict(torch.load(f"{project_path}/best_model_major.pt", map_location=device))
model_major.eval()

# Define prediction function
def predict(sentence):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=64).to(device)

    with torch.no_grad():
        # Get prediction probabilities from both models
        probs_sub = F.softmax(model_sub(inputs["input_ids"], inputs["attention_mask"]), dim=1).squeeze()
        probs_major = F.softmax(model_major(inputs["input_ids"], inputs["attention_mask"]), dim=1).squeeze()

    # Get top 3 predictions from each model
    top3_sub = torch.topk(probs_sub, k=3)
    top3__major = torch.topk(probs_major[:6], k=3)  # Only use first 6 indices for major classes

    result = []
    for i in range(3):
        idx = top3_sub.indices[i].item()
        code = inv_label_mapping60[idx]  # Fine-grained emotion code
        name = emotion_names_sub.get(code, "Unknown")
        score = top3_sub.values[i].item()
        group = coarse_map.get(code, "")  # Coarse-grained emotion code
        
        # Filter predictions: include only if major emotion category also appears in major top-3
        if group in [inv_label_mapping_6[i.item()] for i in top3__major.indices]:
            result.append((group, code, name, score))

    # Display results
    print(f"📝 Input Sentence: {sentence}")
    if result:
        print("🎯 Filtered Emotions:")
        for group, name, score in sorted(result, key=lambda x: -x[3]):
            emoji = emotion_emojis.get(group, "")
            print(f"{name} {emoji} → Probability: {score:.4f}")
    else:
        # If no coarse-grained label matches, display top-3 fine-grained predictions only
        print("⚠️ No matching major category (Top 3 fine-grained predictions):")
        for i in range(3):
            idx = top3_sub.indices[i].item()
            code = inv_label_mapping60[idx]
            name = emotion_names_sub.get(code, "Unknown")
            score = top3_sub.values[i].item()
            print(f"{code} ({name}) - Probability: {score:.4f}")

# Create input field widget for user to enter sentence
text_input = widgets.Text(description="문장:") 

# Output area to display prediction results
output_area = widgets.Output()

# Event handler for input submission
def on_submit(text_widget):
    sentence = text_widget.value.strip()
    if sentence:
        with output_area:
            clear_output()
            predict(sentence)
        text_input.value = ""  # Clear the input field after submission

# Bind the submit event to the input widget and display the interface
text_input.on_submit(on_submit)
display(text_input, output_area)
