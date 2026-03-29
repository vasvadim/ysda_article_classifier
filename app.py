import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Arxiv Classifier", page_icon="📚")
st.title("📚 Классификатор статей ArXiv")

@st.cache_resource
def load_model():
    model_path = "./model"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"Ошибка загрузки: {e}")
    st.stop()

title_input = st.text_input("Название статьи:", placeholder="Attention Is All You Need")
abstract_input = st.text_area("Абстракт (Аннотация):", placeholder="Текст аннотации (опционально)")

if st.button("Квалифицировать статью", type="primary"):
    if not title_input.strip():
        st.warning("Введите название статьи!")
    else:
        try:
            if abstract_input.strip():
                text_to_classify = f"{title_input.strip()}. {abstract_input.strip()}"
            else:
                text_to_classify = title_input.strip()

            inputs = tokenizer(text_to_classify, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).squeeze().tolist()
            
            id2label = model.config.id2label
            class_probs = [(probs[i], id2label[i]) for i in range(len(probs))]
            class_probs.sort(key=lambda x: x[0], reverse=True)
            
            top_95_classes = []
            cumulative_prob = 0.0
            
            for prob, class_name in class_probs:
                top_95_classes.append((class_name, prob))
                cumulative_prob += prob
                if cumulative_prob >= 0.95:
                    break
            
            st.success("Результат:")
            for class_name, prob in top_95_classes:
                st.metric(label=class_name, value=f"{prob * 100:.2f}%")
                st.progress(prob)

        except Exception as e:
            st.error(f"Ошибка при обработке: {e}")