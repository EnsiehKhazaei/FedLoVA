import os
import torch
from federatedscope.core.aggregators import Aggregator
from federatedscope.core.auxiliaries.utils import param2tensor

import copy
import torch.optim as optim
import torch.nn.functional as F
import logging
import math
logger = logging.getLogger(__name__)
class ClientsAvgAggregator(Aggregator):
    """
    Implementation of vanilla FedAvg refer to 'Communication-efficient \
    learning of deep networks from decentralized data' [McMahan et al., 2017] \
    http://proceedings.mlr.press/v54/mcmahan17a.html
    """
    def __init__(self, model=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        self.prev_server_weight = None

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """
        # if self.prev_server_weight:
        #     print(self.prev_server_weight["base_model.model.roberta.encoder.layer.21.attention.self.query.lora_B.default.weight"])
        # print(self.model.state_dict()["base_model.model.roberta.encoder.layer.21.attention.self.query.lora_B.default.weight"])
        # print(dict(self.model.named_parameters())["base_model.model.roberta.encoder.layer.21.attention.self.query.lora_B.default.weight"])
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)  ### Commented only when running LoRA-FAIR
        self.prev_server_weight = copy.deepcopy(avg_model)  ### is added for DP
        # print(avg_model["base_model.model.roberta.encoder.layer.21.attention.self.query.lora_B.default.weight"])
        # ################################## Added for LoRA-FAIR #######################################
        # # sample_size, A = models[0]
        # # print('before averaging')
        # # print(A['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_A.default.weight'])
        # # print(A['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_B.default.weight'])
        # # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.value.lora_A.default.weight'])
        # # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.value.lora_B.default.weight'])
        
        # avg_lora_A, avg_lora_B, avg_model, avg_products = self._para_weighted_avg2(models, recover_fun=recover_fun)
        # # # sample_size, A = models[0]
        # # print('after averaging')
        # # print(models[0][1]['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_A.default.weight'])
        # # print(models[0][1]['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_B.default.weight'])
        # # print("**************** Aggregation Happens Here ******************************************")
        # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_A.default.weight'])
        # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_B.default.weight'])
        
        # delta_lora_B = self._optimize_delta_lora_B(avg_lora_A, avg_lora_B, avg_products, num_iterations=600, learning_rate=0.01, lambda_reg=1.0)

        # # avg_model = models[0][1]     
        # # # Average all parameters
        # for key in avg_model.keys():
        #     # print(torch.norm(avg_lora_B[key] - models[0][1][key]))
        #     if key in delta_lora_B:
        #             # print(key)
        #             avg_model[key] += delta_lora_B[key]
        #             # print(key)
        #             # print(torch.norm(delta_lora_B[key]))
        #             # print(torch.norm(avg_model[key]))
        
        # # print('after updating B')
        # # print(models[1][1]['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_A.default.weight'])
        # # print(models[1][1]['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_B.default.weight'])
        # # print("**************** Aggregation Happens Here ******************************************")
        # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_A.default.weight'])
        # # print(avg_model['base_model.model.roberta.encoder.layer.9.attention.self.query.lora_B.default.weight'])
        
        return avg_model

    def update(self, model_parameters):
        """
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        self.model.load_state_dict(model_parameters, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, path)

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))


    def _dp_sigma_from_budget(self, clip_norm, epsilon, delta):
        """
        Analytic calibration of Gaussian mechanism noise std from
        (epsilon, delta)-DP budget, using the standard bound:
        sigma >= C * sqrt(2 * ln(1.25 / delta)) / epsilon
        This is per-round, per-client sensitivity C = clip_norm.
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError(
                f"Invalid DP budget: epsilon={epsilon}, delta={delta}. "
                f"Require epsilon > 0 and 0 < delta < 1.")
        return (clip_norm * math.sqrt(2.0 * math.log(1.25 / delta))) / epsilon

    def _clip_and_noise_delta(self, local_model, clip_norm, epsilon, delta,
                           keys=None):
        """
        Computes delta = local_model - self.prev_server_weight (restricted to
        `keys`, i.e. the parameters actually shared with the server), clips
        its global L2 norm to `clip_norm`, and adds Gaussian noise calibrated
        to (epsilon, delta)-DP.

        Returns a dict of noised deltas for `keys` only.
        """
        keys = keys if keys is not None else list(local_model.keys())
        sigma = self._dp_sigma_from_budget(clip_norm, epsilon, delta)
        
        deltas = {}
        total_norm_sq = 0.0
        for key in keys:
            local_tensor = param2tensor(local_model[key]).float()
            prev_tensor = param2tensor(self.prev_server_weight[key]).float()
            # Make sure both operands are on the same device before subtracting
            prev_tensor = prev_tensor.to(local_tensor.device)
            d = local_tensor - prev_tensor
            deltas[key] = d
            total_norm_sq += torch.sum(d ** 2).item()
        total_norm = total_norm_sq ** 0.5
        clip_coef = min(1.0, clip_norm / (total_norm + 1e-6))
        print(total_norm)
        noised_deltas = {}
        for key in keys:
            clipped = deltas[key] * clip_coef
            noise = torch.normal(mean=0.0, std=sigma, size=clipped.shape, device=clipped.device, dtype=clipped.dtype)
            noised_deltas[key] = clipped + noise
        return noised_deltas

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size
        
        #################### Added only for DP noise ###############################
        # if self.prev_server_weight:
        #     clip_norm = 0.005
        #     epsilon = 50.0
        #     delta = 1e-5
            
        #     dp_models = []
        #     for sample_size, local_model in models:
        #         keys = list(local_model.keys())
        #         noised_deltas = self._clip_and_noise_delta(
        #             local_model, clip_norm, epsilon, delta, keys=keys)
        #         new_local_model = dict(local_model)
        #         for key in keys:
        #             prev_tensor = param2tensor(self.prev_server_weight[key]).float()
        #             # new_local_model[key] = torch.nan_to_num(prev_tensor + noised_deltas[key], nan=0.0, posinf=1e3, neginf=-1e3)
        #             new_local_model[key] = prev_tensor + noised_deltas[key]
        #         dp_models.append((sample_size, new_local_model))
        #     models = dp_models
    # If use_dp is False (including the prev_server_weight is None case),
    # `models` is left untouched and we fall through to plain averaging.
        #################### Added for DP noise ###############################
        
        sample_size, avg_model = models[0]
        # avg_model = copy.deepcopy(models[0][1])
        for key in avg_model:
            # print(key)
            for i in range(len(models)):
                local_sample_size, local_model = models[i]
                
                if self.cfg.federate.ignore_weight:
                    weight = 1.0 / len(models)
                elif self.cfg.federate.use_ss:
                    # When using secret sharing, what the server receives
                    # are sample_size * model_para
                    weight = 1.0
                else:
                    weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model
    def _para_weighted_avg2(self, models, recover_fun=None):
        """
        Calculates the weighted average of models,
        and additionally computes the weighted average of LoRA products (B x A).
        """
        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        # avg_model = copy.deepcopy(first_model)
        avg_products = {}  # stores weighted avg of B x A per lora_A key
        avg_lora_A = {}
        avg_lora_B = {}
        
        for i in range(len(models)):
            local_sample_size, local_model = models[i]

            if self.cfg.federate.ignore_weight:
                weight = 1.0 / len(models)
            elif self.cfg.federate.use_ss:
                weight = 1.0
            else:
                weight = local_sample_size / training_set_size

            for key in local_model:
                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                if 'lora_A' in key:
                    lora_A = param2tensor(local_model[key])
                    lora_B = param2tensor(local_model[key.replace('lora_A', 'lora_B')])
                    # Flatten for matrix multiplication
                    lora_A_flat = lora_A.view(lora_A.size(0), -1).float()
                    lora_B_flat = lora_B.view(lora_B.size(0), -1).float()
                    
                    # Compute product
                    if lora_B_flat.size(1) != lora_A_flat.size(0):
                        raise ValueError(f"Dimension mismatch: LoRA_B {lora_B_flat.size()} vs LoRA_A {lora_A_flat.size()}")
                    
                    product = torch.matmul(lora_B_flat, lora_A_flat)
                    # Accumulate
                    if key not in avg_lora_A:
                        avg_lora_A[key] = lora_A.clone() * weight
                        avg_lora_B[key.replace('lora_A', 'lora_B')] = lora_B.clone() * weight
                        avg_products[key] = product * weight
                    else:
                        avg_lora_A[key] += lora_A * weight
                        avg_lora_B[key.replace('lora_A', 'lora_B')] += lora_B * weight
                        avg_products[key] += product * weight
        for key in avg_lora_A:
            avg_model[key] = copy.deepcopy(avg_lora_A[key])

        for key in avg_lora_B:
            avg_model[key] = copy.deepcopy(avg_lora_B[key])
        # for key in avg_products:
        #     print(key)
        # print(avg_products[key])
        return avg_lora_A, avg_lora_B, avg_model, avg_products
    
    def _optimize_delta_lora_B(self, hat_lora_A, hat_lora_B, avg_client_product, num_iterations=10, learning_rate=0.01, lambda_reg=1.0):
        """
        Optimize delta_LoRA_B to minimize reconstruction error.
        
        This function optimizes correction terms for LoRA_B parameters to minimize
        the difference between averaged client products and reconstructed products.
        
        Args:
            hat_lora_A (dict): Averaged LoRA_A parameters
            hat_lora_B (dict): Averaged LoRA_B parameters  
            avg_client_product (dict): Averaged client LoRA products
            num_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimization
            lambda_reg (float): Regularization weight
        """
        delta_lora_B_dict = {}
        optimizer_params = []

        # Initialize delta_LoRA_B for each layer
        for key in hat_lora_B.keys():
            # Create delta_LoRA_B tensor with Xavier initialization
            delta_lora_B = torch.empty_like(hat_lora_B[key], dtype=torch.float32)
            torch.nn.init.xavier_uniform_(delta_lora_B)
            delta_lora_B.requires_grad = True
            
            delta_lora_B_dict[key] = delta_lora_B
            optimizer_params.append(delta_lora_B)

        # Setup optimizer
        optimizer = optim.SGD(optimizer_params, lr=learning_rate)

        # Optimization loop for delta_LoRA_B
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0
            loss_term1 = 0
            loss_term2 = 0
            # Compute loss for each layer
            for key in delta_lora_B_dict.keys():
                # Construct corrected LoRA_B
                corrected_lora_B = hat_lora_B[key] + delta_lora_B_dict[key]

                # Compute objective: || avg(LoRA_A * LoRA_B) - hat(LoRA_A) * corrected_LoRA_B ||
                lora_A_flattened = hat_lora_A[key.replace('lora_B', 'lora_A')].view(
                    hat_lora_A[key.replace('lora_B', 'lora_A')].size(0), -1).float()
                corrected_lora_B_flattened = corrected_lora_B.view(corrected_lora_B.size(0), -1).float()

                if corrected_lora_B_flattened.size(1) != lora_A_flattened.size(0):
                    raise ValueError(f"Dimension mismatch: corrected_LoRA_B {corrected_lora_B_flattened.size()} vs hat_LoRA_A {lora_A_flattened.size()}")

                reconstructed_product = torch.matmul(corrected_lora_B_flattened, lora_A_flattened)
                
                # Use cosine similarity loss
                # print(key)
                # print("x norm:", torch.norm(x))
                # print("y norm:", torch.norm(y))
                cosine_similarity = F.cosine_similarity(
                    avg_client_product[key.replace('lora_B', 'lora_A')], 
                    reconstructed_product, dim=0)
                # mse_loss = F.mse_loss(
                #     avg_client_product[key.replace('lora_B', 'lora_A')], 
                #     reconstructed_product, dim=0)
                # if torch.isnan(cosine_similarity).any():
                #     print("NaN in cosine similarity at key:", key)
                loss_term1 += 1 - cosine_similarity.mean()
                # loss_term1 += F.mse_loss(avg_client_product[key.replace('lora_B', 'lora_A')], 
                #     reconstructed_product)
                # loss_term1 = mse_loss.mean()

                # Regularization term: lambda * ||delta_LoRA_B||
                loss_term2 += lambda_reg * torch.norm(delta_lora_B_dict[key]) ** 2

                # Total loss for this layer
                total_loss = loss_term1 + loss_term2

            # Backpropagation and optimization
            total_loss.backward()
            optimizer.step()
            
            # Optional: print progress
            if iteration % 100 == 0:
                print(f'Iteration {iteration}, Loss term 1: {loss_term1:.10f}, Loss term 2: {loss_term2:.10f}')
                logger.info(f'Iteration {iteration}, Loss term 1: {loss_term1:.6f}, Loss term 2: {loss_term2:.6f}')
                # print(f'Iteration {iteration}, Loss term 1: {loss_term1:.6f}, Loss term 2: {loss_term2:.6f}')
                # logger.info(f'Iteration {iteration}, Loss term 1: {loss_term1:.6f}, Loss term 2: {loss_term2:.6f}')

        # Store optimized delta_LoRA_B
        delta_lora_B = {key: delta_lora_B_dict[key].detach() for key in delta_lora_B_dict.keys()}
        return delta_lora_B


class OnlineClientsAvgAggregator(ClientsAvgAggregator):
    """
    Implementation of online aggregation of FedAvg.
    """
    def __init__(self,
                 model=None,
                 device='cpu',
                 src_device='cpu',
                 config=None):
        super(OnlineClientsAvgAggregator, self).__init__(model, device, config)
        self.src_device = src_device

    def reset(self):
        """
        Reset the state of the model to its initial state
        """
        self.maintained = self.model.state_dict()
        for key in self.maintained:
            self.maintained[key].data = torch.zeros_like(
                self.maintained[key], device=self.src_device)
        self.cnt = 0

    def inc(self, content):
        """
        Increment the model weight by the given content.
        """
        if isinstance(content, tuple):
            sample_size, model_params = content
            for key in self.maintained:
                # if model_params[key].device != self.maintained[key].device:
                #    model_params[key].to(self.maintained[key].device)
                self.maintained[key] = (self.cnt * self.maintained[key] +
                                        sample_size * model_params[key]) / (
                                            self.cnt + sample_size)
            self.cnt += sample_size
        else:
            raise TypeError(
                "{} is not a tuple (sample_size, model_para)".format(content))

    def aggregate(self, agg_info):
        """
        Returns the aggregated value
        """
        return self.maintained
