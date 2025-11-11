import os
import argparse
from ruamel.yaml import YAML
from transformers.models.auto.image_processing_auto import image_processors
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
from transformers import AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

import yaml
import torch
import json
import copy
import gc
import pickle
import random
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from PIL import Image
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
# from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers.cache_utils import Cache, StaticCache
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast




parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/config.yaml', help='global environment configs')
args = parser.parse_args()
yaml = YAML()

# Reading a YAML file
with open(args.config, 'r') as file:
    config = yaml.load(file)
    print(config)

DATA_NAME = 'm3cot'
MODEL_TYPE = 'qwen'
IMG_FOLDER = config['dataset'][DATA_NAME]['IMG_FOLDER']
EVAL_FILE = config['dataset'][DATA_NAME]['EVAL_FILE']
prob_diff_threshold = config['model'][MODEL_TYPE]['prob_diff_threshold']
num_selected_patches = config['model'][MODEL_TYPE]['num_selected_patches']
MODEL_PATH = config['model'][MODEL_TYPE]['model_path']
MCOT = config['MCOT']


@dataclass
class Qwen2VLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    selected_image_embeddings: Optional[torch.FloatTensor] = None
    selected_vokens: torch.LongTensor = None


class Qwen2VLForInterCoT(Qwen2VLForConditionalGeneration):
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            sub_image_masks=None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            image_pool: Optional[list] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to("cuda")
            if pixel_values is None and self.be_updated is True:
                inputs_embeds[:, -self.last_length + 1:-2, :] = self.selected_voken.unsqueeze(0)
                self.be_updated = False


            if pixel_values is not None:
                self.image_start_id = 151652
                self.image_end_id = 151653
                self.be_updated = False
                self.last_length = 0
                self.num_line_break = 0
                pixel_values = torch.tensor(pixel_values).type(self.visual.get_dtype()).to("cuda")
                image_grid_thw = torch.tensor(image_grid_thw, device="cuda", dtype=torch.long)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                my_grid = torch.tensor(image_grid_thw)
                cu_seqlens = torch.repeat_interleave(my_grid[:, 1] * my_grid[:, 2], my_grid[:, 0]).cumsum(
                    dim=0, dtype=torch.int32
                )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                cu_seqlens = cu_seqlens // 4
                self.voken_start, self.voken_end = cu_seqlens[-2], cu_seqlens[-1]
                self.period_token_id = 13  # ID of '.'

                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                self.query_vokens = image_embeds[self.voken_start:self.voken_end, :]


            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)


        # inilize for kv cot
        if output_attentions and past_key_values.key_cache == [] and pixel_values is not None:
            self.new_tokens = 0
            self.num_selected_patches = num_selected_patches

            self.image_pool = image_pool if image_pool is not None else []
            self.select_from_pool_config = bool(self.image_pool)

            self.current_sentence_prob_diff_sum = 0.0
            self.current_sentence_token_count = 0

            self.query_image_mask = torch.zeros_like(input_ids, device=input_ids.device).bool()
            self.start_idx = (input_ids == 151652).nonzero(as_tuple=True)[1].max().item() + 1
            self.end_idx = (input_ids == 151653).nonzero(as_tuple=True)[1].max().item()
            self.query_image_mask[:, self.start_idx:self.end_idx] = True


        elif output_attentions:
            self.new_tokens += 1
            false_tensor = torch.tensor([[False]], device=self.query_image_mask.device)
            self.query_image_mask = torch.cat([self.query_image_mask, false_tensor], dim=-1)



        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)


        complexity = False

        if pixel_values is None and input_ids is not None:
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k=2)
            prob_diff = (top_k_probs[:, 0] - top_k_probs[:, 1]).item()
            self.current_sentence_prob_diff_sum += prob_diff
            self.current_sentence_token_count += 1

            if input_ids[:, -1].item() == self.period_token_id:
                if self.current_sentence_token_count > 0:
                    average_prob_diff = self.current_sentence_prob_diff_sum / self.current_sentence_token_count
                    response_list.append(average_prob_diff)

                    if average_prob_diff < prob_diff_threshold:
                        complexity = True

                self.current_sentence_prob_diff_sum = 0.0
                self.current_sentence_token_count = 0

        selected_vokens = None
        if output_attentions and input_ids[:, -1] == 1699:
            self.num_line_break += 1

        self.select_from_pool = True

        insert_condition = output_attentions and complexity and input_ids[:, -1] == self.period_token_id

        if insert_condition and self.select_from_pool and image_pool != []:
            mask_scores = []
            valid_masks_indices = []
            image_attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]

            for i, mask_info in enumerate(self.image_pool):
                relative_indices = mask_info['vq_token_indices'].to(input_ids.device)
                absolute_indices = relative_indices + self.start_idx
                attention_for_mask = image_attentions[:,absolute_indices]
                mask_score = torch.sum(attention_for_mask).item()
                mask_scores.append(mask_score)
                valid_masks_indices.append(i)

            if mask_scores:
                best_score_index_in_valid = torch.argmax(torch.tensor(mask_scores)).item()
                best_mask_original_index = valid_masks_indices[best_score_index_in_valid]
                best_mask_info = self.image_pool[best_mask_original_index]
                best_mask_score = mask_scores[best_score_index_in_valid]
                print(f"Selected mask index {best_mask_original_index} ('{best_mask_info.get('mask_id', 'N/A')}') with score {best_mask_score:.4f}")

                selected_relative_indices = torch.sort(best_mask_info['vq_token_indices'].to(input_ids.device))[0]
                num_inserted_image_tokens = len(selected_relative_indices)
                selected_vokens = self.query_vokens[selected_relative_indices, :]
                self.image_pool.pop(best_mask_original_index)
                num_to_pad = num_inserted_image_tokens
                self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                      num_to_pad + 2,
                                                                                      device=self.query_image_mask.device).bool()],dim=1)
                self.be_updated = True
                self.last_length = len(selected_vokens) + 3
                self.selected_voken = selected_vokens


        elif insert_condition and self.select_from_pool is False:
            image_attentions = torch.cat(outputs.attentions, dim=1).mean(dim=1)[:, -1]
            image_attentions = image_attentions[self.query_image_mask]
            selected_patches = image_attentions.topk(self.num_selected_patches)[1]
            selected_patches = sorted((selected_patches).tolist())
            selected_vokens = self.query_vokens[selected_patches, :]
            selected_patches = torch.tensor(selected_patches, device=input_ids.device) + self.start_idx
            self.query_image_mask[:, selected_patches] = False
            self.query_image_mask = torch.cat([self.query_image_mask, torch.zeros(self.query_image_mask.shape[0],
                                                                                  selected_vokens.shape[0] + 2,
                                                                                  device=self.query_image_mask.device).bool()],dim=1)
            self.be_updated = True
            self.last_length = len(selected_vokens) + 3
            self.selected_voken = selected_vokens

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
            selected_vokens=selected_vokens,
        )



    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                if 'selected_vokens' in kwargs and kwargs['selected_vokens'] is not None:
                    input_ids = input_ids[:, cache_position[-1] - (kwargs['selected_vokens'].shape[0] + 2):]
                else:
                    input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
                device = inputs_embeds.device
            else:
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            attention_mask = self.model._prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.lm_head.weight.dtype,
                device=device,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )

        if 'sub_image_masks' in kwargs:
            model_inputs.update(
                {
                    "sub_image_masks": kwargs['sub_image_masks'],
                }
            )

        if 'image_pool' in kwargs:
            model_inputs.update(
                {
                    "image_pool": kwargs['image_pool']
                }
            )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:

        cache_name, cache = self._extract_past_from_model_output(outputs)
        model_kwargs[cache_name] = cache

        if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
            model_kwargs['selected_vokens'] = outputs['selected_vokens']
        elif 'selected_vokens' in model_kwargs:
            model_kwargs.pop('selected_vokens')


        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
                if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
                    # model_kwargs["attention_mask"] = torch.cat(
                    #     [model_kwargs["attention_mask"], torch.ones_like(outputs['selected_vokens'])], dim=-1)
                    append_len = outputs['selected_vokens'].shape[0] + 2
                    append_mask = torch.ones(1, append_len, dtype=model_kwargs["attention_mask"].dtype,
                                             device=model_kwargs["attention_mask"].device)
                    model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], append_mask], dim=-1)
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )
        # TODO: cache_position is not applied to the prefix vokens
        if model_kwargs.get("use_cache", True):
            if 'selected_vokens' in outputs and outputs['selected_vokens'] is not None:
                num_new_tokens += outputs['selected_vokens'].shape[0] + 2
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            past_positions = model_kwargs.pop("cache_position")
            new_positions = torch.arange(
                past_positions[-1] + 1, past_positions[-1] + num_new_tokens + 1, dtype=past_positions.dtype
            ).to(past_positions.device)
            model_kwargs["cache_position"] = torch.cat((past_positions, new_positions))

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs



dataset = open(EVAL_FILE).readlines()
dataset = [json.loads(d) for d in dataset]
dataset = [x for x in dataset if x['image'] is not None]
model = Qwen2VLForInterCoT.from_pretrained(MODEL_PATH, attn_implementation=config['attn']).to(device='cuda', dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

generation_config = {
    'do_sample': True,
    'temperature': 0.7,
    'top_p': 0.9,
    'repetition_penalty': 1.2,
    'min_new_tokens': 128,
    'max_new_tokens': 2048
}



def calculate_generated_text(prompt, vision_x, image_pool):
    """
    Calculate generated text given a prompt and vision data.

    Parameters:
    - prompt (str): The input prompt.
    - vision_x (list[PIL Images]): List of PIL Images containing vision data.

    Returns:
    Tuple[str, str]: Tuple containing the raw and salt answer text.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": vision_x[0],
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # Preprocess the inputs

    inputs = processor(
        text=[text], images=[image_inputs], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    inputs['output_attentions'] = MCOT
    inputs['image_pool'] = image_pool

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, **generation_config)
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    return output_text[0]


zero_shot_prompt_template = '''Question: {}
Options:
'''

output_format = """
Output Format:
<<<
After reasoning, you **MUST give your final answer** using the **EXACT format** below (including 'Answer:' and the option letter):

**Answer: [Your Final Option]**

For example:
[Here is your reasoning text].
**Answer: B**
>>>"""

def open_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    image_pool_path = f"data_all/{DATA_NAME}/{DATA_NAME}_sam2_image_pool_{MODEL_TYPE}.pkl"

    image_pool = open_pkl(image_pool_path)
    output_dir = './results/{}'.format(DATA_NAME)
    os.makedirs(output_dir, exist_ok=True)

    mcot_zero_fh = open(output_dir + '/qwen_mcot_zero' + str(prob_diff_threshold) + '.json'.format(DATA_NAME), 'a')

    for idx, data in enumerate(tqdm(dataset)):
        try:
            print("="*200)
            mcot_input_str = zero_shot_prompt_template.format(data['question'])
            for i, c in zip(['A', 'B', 'C', 'D', 'E', 'F'], data['choices']):
                mcot_input_str += '{}. {}\n'.format(i, c)

            zero_shot_vision = [os.path.join(IMG_FOLDER, data['image_id']+'.png')]

            zero_shot_mcot_input_str = mcot_input_str + '\n' + output_format

            id = data['id']

            zero_shot = calculate_generated_text(zero_shot_mcot_input_str, zero_shot_vision, image_pool[id])
            zeroshot_mcot_output = copy.deepcopy(data)
            zeroshot_mcot_output['pred'] = zero_shot
            mcot_zero_fh.write(json.dumps(zeroshot_mcot_output) + '\n')
            print(f"zeroshot_mcot_output:\n{zeroshot_mcot_output}\n")

            del zero_shot, zero_shot_vision

            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"eeee:{e}")



if __name__ == '__main__':
    main()