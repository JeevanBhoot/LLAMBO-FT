import os
import time
import re
import numpy as np
from scipy.stats import norm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llambo.rate_limiter import RateLimiter
from llambo.discriminative_sm_utils import gen_prompt_templates



class LLM_DIS_SM_HuggingFace:
    def __init__(self, task_context, n_gens, lower_is_better,
                model_path="meta-llama/Meta-Llama-3-8B-Instruct",
                bootstrapping=False, n_templates=1, 
                use_recalibration=False,
                rate_limiter=None, warping_transformer=None,
                verbose=False, prompt_setting=None, shuffle_features=False):
        '''Initialize the forward LLM surrogate model. This is modelling p(y|x) as in GP/SMAC etc.'''
        self.task_context = task_context
        self.n_gens = n_gens
        self.lower_is_better = lower_is_better
        self.bootstrapping = bootstrapping
        self.n_templates = n_templates
        assert not (bootstrapping and use_recalibration), 'Cannot do recalibration and boostrapping at the same time' 
        self.use_recalibration = use_recalibration
        if rate_limiter is None:
            self.rate_limiter = RateLimiter(max_tokens=100000, time_frame=60)
        else:
            self.rate_limiter = rate_limiter
        if warping_transformer is not None:
            self.warping_transformer = warping_transformer
            self.apply_warping = True
        else:
            self.warping_transformer = None
            self.apply_warping = False
        self.recalibrator = None
        self.verbose = verbose
        self.prompt_setting = prompt_setting
        self.shuffle_features = shuffle_features

        assert type(self.shuffle_features) == bool, 'shuffle_features must be a boolean'

        # Load the model and tokenizer from Huggingface
        self.model, self.tokenizer = self.load_model(model_path)

    def load_model(self, model_path):
        '''Load the Huggingface model.'''
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        return model, tokenizer

    def _generate_response(self, input_text, n_preds=3):
        '''Generate response from the Huggingface model.'''
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that helps people find information."
                ),
            },
            {"role": "user", "content": input_text},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = []
        for _ in range(self.n_gens):
            outputs.extend(self.model.generate(
                input_ids,
                max_new_tokens=20,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
            ))

        responses = [
            self.tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True)
            for output in outputs
        ]
        print(responses)
        print(len(responses))
        return responses

    def _generate(self, few_shot_template, query_example):
        '''Generate multiple responses from the Huggingface model for Monte Carlo approach.'''
        user_message = few_shot_template.format(Q=query_example['Q'])
        responses = self._generate_response(user_message, n_preds=max(self.n_gens, 3))

        # Extract numerical values from responses using regex, adapting to patterns with or without hashtags
        final_responses = []
        for response in responses:
            match = re.findall(r"## (-?[\d.]+) ##", response)
            # Try to match pattern: '## <value> ##
            if match:
                final_responses.append(float(match[-1]))
            else:
                # Sometimes LLM just produces <value> without ## ## pattern
                try:
                    final_responses.append(float(response))
                except ValueError:
                    final_responses.append(np.nan)

        tot_tokens = len(user_message.split()) + sum(len(str(response).split()) for response in responses if response is not None)
        tot_cost = 0  # No cost associated with using local model

        return final_responses, tot_cost, tot_tokens

    def _predict(self, all_prompt_templates, query_examples):
        start = time.time()
        all_preds = []
        tot_tokens = 0
        tot_cost = 0

        for i, query in enumerate(query_examples):
            query_preds = []
            for template in all_prompt_templates:
                responses, cost, tokens = self._generate(template, query)
                query_preds.extend(responses)
                while len(query_preds) < self.n_gens:
                    query_preds.append(np.nan)
                tot_cost += cost
                tot_tokens += tokens
            all_preds.append(query_preds)
        
        end = time.time()
        time_taken = end - start

        all_preds = np.array(all_preds).astype(float)
        y_mean = np.nanmean(all_preds, axis=1)
        y_std = np.nanstd(all_preds, axis=1)

        # Capture failed calls - impute None with average predictions
        y_mean[np.isnan(y_mean)]  = np.nanmean(y_mean)
        y_std[np.isnan(y_std)]  = np.nanmean(y_std)
        y_std[y_std<1e-5] = 1e-5  # replace small values to avoid division by zero

        return y_mean, y_std, tot_cost, tot_tokens, time_taken

    def _evaluate_candidate_points(self, observed_configs, observed_fvals, candidate_configs, 
                                   use_context='full_context', use_feature_semantics=True, return_ei=False):
        '''Evaluate candidate points using the LLM model.'''

        if self.prompt_setting is not None:
            use_context = self.prompt_setting

        all_run_cost = 0
        all_run_time = 0

        tot_cost = 0
        time_taken = 0

        if self.use_recalibration and self.recalibrator is None:
            recalibrator, tot_cost, time_taken = self._get_recalibrator(observed_configs, observed_fvals)
            if recalibrator is not None:
                self.recalibrator = recalibrator
            else:
                self.recalibrator = None
            print(f'[Recalibration] COMPLETED')

        all_run_cost += tot_cost
        all_run_time += time_taken

        all_prompt_templates, query_examples = gen_prompt_templates(self.task_context, observed_configs, observed_fvals, candidate_configs, 
                                                                    n_prompts=self.n_templates, bootstrapping=self.bootstrapping,
                                                                    use_context=use_context, use_feature_semantics=use_feature_semantics, 
                                                                    shuffle_features=self.shuffle_features, apply_warping=self.apply_warping)

        print('*'*100)
        print(f'Number of all_prompt_templates: {len(all_prompt_templates)}')
        print(f'Number of query_examples: {len(query_examples)}')
        print(all_prompt_templates[0].format(Q=query_examples[0]['Q']))

        response = self._predict(all_prompt_templates, query_examples)

        y_mean, y_std, tot_cost, tot_tokens, time_taken = response

        if self.recalibrator is not None:
            recalibrated_res = self.recalibrator(y_mean, y_std, 0.68)   # 0.68 coverage for 1 std
            y_std = np.abs(recalibrated_res.upper - recalibrated_res.lower)/2

        all_run_cost += tot_cost
        all_run_time += time_taken

        if not return_ei:
            return y_mean, y_std, all_run_cost, all_run_time
    
        else:
            # calculate ei
            if self.lower_is_better:
                best_fval = np.min(observed_fvals.to_numpy())
                delta = -1*(y_mean - best_fval)
            else:
                best_fval = np.max(observed_fvals.to_numpy())
                delta = y_mean - best_fval
            with np.errstate(divide='ignore'):  # handle y_std=0 without warning
                Z = delta/y_std
            ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

            return ei, y_mean, y_std, all_run_cost, all_run_time

    def select_query_point(self, observed_configs, observed_fvals, candidate_configs):
        '''Select the next query point using expected improvement.'''

        # warp
        if self.warping_transformer is not None:
            observed_configs = self.warping_transformer.warp(observed_configs)
            candidate_configs = self.warping_transformer.warp(candidate_configs)

        y_mean, y_std, cost, time_taken = self._evaluate_candidate_points(observed_configs, observed_fvals, candidate_configs)
        if self.lower_is_better:
            best_fval = np.min(observed_fvals.to_numpy())
            delta = -1*(y_mean - best_fval)
        else:
            best_fval = np.max(observed_fvals.to_numpy())
            delta = y_mean - best_fval

        with np.errstate(divide='ignore'):  # handle y_std=0 without warning
            Z = delta/y_std

        ei = np.where(y_std>0, delta * norm.cdf(Z) + y_std * norm.pdf(Z), 0)

        best_point_index = np.argmax(ei)

        # unwarp
        if self.warping_transformer is not None:
            candidate_configs = self.warping_transformer.unwarp(candidate_configs)

        best_point = candidate_configs.iloc[[best_point_index], :]  # return selected point as dataframe not series
        return best_point, cost, time_taken
