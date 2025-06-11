from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
import torch
import warnings
import torch.nn.functional as F

warnings.filterwarnings("ignore")

class LLava:
    def __init__(self, pretrained, model_name, tempurature=0):
        
        # llava-next-interleave-7b
        # llava-onevision-qwen2-7b-ov
        self.pretrained = f"lmms-lab/{pretrained}"
        self.model_name = model_name
        self.device = "cuda"
        self.device_map = "auto"
        self.llava_model_args = {
            "multimodal": True,
        }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        self.llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, model_name, device_map=self.device_map, **self.llava_model_args)
        self.tempurature = tempurature
        self.model.eval()
    
    def reload(self):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.pretrained, None, self.model_name, device_map=self.device_map, **self.llava_model_args)
        self.model.eval()
        
    
    
    def __call__(self, qs, img_files, num_return_sequences=1, do_sample=False, temperature=0, reload=False):
        # reload_llm
        if reload == True:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.reload()
        
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        
        if img_files:
            image_tensors = process_images(img_files, self.image_processor, self.model.config)
            image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]
            image_sizes = [image.size for image in img_files]
        else:
            image_tensors = None
            image_sizes = None
        
        with torch.inference_mode():
            cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=4096,
            num_return_sequences=num_return_sequences,
        )

        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        outputs = text_outputs
        return outputs
    
    
    
    def compute_log_prob(self, question, imgs, answer):
        # Step 1: Prepare prompt
        conv = copy.deepcopy(conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Step 2: Tokenize prompt and answer
        prompt_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False, return_tensors="pt").to(self.device)

        # Step 3: Process image
        image_tensors = process_images(imgs, self.image_processor, self.model.config)
        image_tensors = [img.to(dtype=torch.float16, device=self.device) for img in image_tensors]
        image_sizes = [img.size for img in imgs]

        # Step 4: Concatenate prompt + answer tokens
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1)

        # Step 5: Forward pass (disable gradients)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                return_dict=True
            )
            logits = outputs.logits[0]  # Shape: [seq_len, vocab_size]
        print(f"Logits shape: {outputs.logits}")
        # Step 6: Get logits corresponding to the answer tokens
        prompt_len = prompt_ids.shape[1]
        answer_logits = logits[prompt_len-1 : prompt_len-1 + answer_ids.shape[1]]

        log_probs = F.log_softmax(answer_logits, dim=-1)
        answer_tokens = answer_ids.squeeze(0)  # Shape: [answer_len]
        token_log_probs = log_probs.gather(1, answer_tokens.unsqueeze(1)).squeeze(1)

        total_log_prob = token_log_probs.sum()
        probability = torch.exp(total_log_prob).item()
        return probability

        
