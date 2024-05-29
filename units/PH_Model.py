import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, PegasusTokenizer

class T5ParaphrasePaws:
    def __init__(self, model_name="Vamsi/T5_Paraphrase_Paws"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        

    def paraphrase(self, sentence, num_return_sequences=1, max_length=256):
        text = "paraphrase: " + sentence + " </s>"
        encoding = self.tokenizer(text, padding='max_length', max_length=max_length, return_tensors="pt", truncation=True)
        input_ids, attention_masks = encoding["input_ids"].to('cuda'), encoding["attention_mask"].to('cuda')

        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            max_length=max_length,
            do_sample=True,
            top_k=120,
            top_p=0.95,
            early_stopping=True,
            num_return_sequences=num_return_sequences
        )
        return [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

class PegasusParaphrase:
    def __init__(self, model_name='tuner007/pegasus_paraphrase'):
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to("cpu")
        self.torch_device = torch_device

    def paraphrase(self, input_text, num_return_sequences=1, num_beams=5, max_length=60):
        batch = self.tokenizer([input_text], truncation=True, padding='longest', max_length=max_length, return_tensors="pt").to("cpu")
        translated = self.model.generate(
            **batch, 
            max_length=max_length, 
            num_beams=num_beams, 
            num_return_sequences=num_return_sequences, 
            temperature=1.5
        )
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)

    
