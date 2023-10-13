import torch
from torch.nn.parameter import Parameter

class Quentizer():
    def __init__(self, args):
        self.args = args
        self.mean_wh, self.mean_bh, self.std_wh, self.std_bh = 0, 0, 0, 0
        self.mean_wo, self.mean_bo, self.std_wo, self.std_bo = 0, 0, 0, 0
        self.is_entropy = args.is_entropy

    def quantize(self, net, num_bit):
        if self.args.mode in ["coin", "sw"]:
            num_bit_entropy = num_bit
            all_param = torch.tensor([])
            for i, layer in enumerate(net.net):
                weight = layer[0].weight
                norm_weight = (weight - self.mean_wh[i]) / self.std_wh[i]
                norm_weight = torch.clamp(norm_weight, -self.args.std_range, self.args.std_range)
                norm_weight = norm_weight / (2 * self.args.std_range) + 0.5
                num_byte = 2**num_bit
                quantized_weight = torch.round(norm_weight * (num_byte - 1))
                if self.is_entropy:
                    all_param = torch.cat([all_param.int(), quantized_weight.to("cpu").int().view(-1)])
                dequantized_norm_weight = quantized_weight / (num_byte - 1)
                dequantized_norm_weight = (dequantized_norm_weight - 0.5) * 2 * self.args.std_range
                dequantized_weight = dequantized_norm_weight * self.std_wh[i] + self.mean_wh[i]
                layer[0].weight = Parameter(dequantized_weight)
            if self.is_entropy:
                train_dist = self.compute_frequency_distribution(all_param, num_bit)
                num_bit_entropy = self.cross_entropy(train_dist, train_dist)
                return net, num_bit_entropy
            return net, num_bit_entropy
        
        elif self.args.mode in ["sd", "swd"]:
            all_param = torch.tensor([])
            for i, layer in enumerate(net.hid_net):
                weight = layer[0].weight
                norm_weight = (weight - self.mean_wh[i]) / self.std_wh[i]
                norm_weight = torch.clamp(norm_weight, -self.args.std_range, self.args.std_range)
                norm_weight = norm_weight / (2 * self.args.std_range) + 0.5
                num_byte = 2**num_bit
                quantized_weight = torch.round(norm_weight * (num_byte - 1))
                if self.is_entropy:
                    all_param = torch.cat([all_param.int(), quantized_weight.to("cpu").int().view(-1)])
                dequantized_norm_weight = quantized_weight / (num_byte - 1)
                dequantized_norm_weight = (dequantized_norm_weight - 0.5) * 2 * self.args.std_range
                dequantized_weight = dequantized_norm_weight * self.std_wh[i] + self.mean_wh[i]
                layer[0].weight = Parameter(dequantized_weight)
            for i, layer in enumerate(net.out_net):
                weight = layer[0].weight
                norm_weight = (weight - self.mean_wo[i]) / self.std_wo[i]
                norm_weight = torch.clamp(norm_weight, -self.args.std_range, self.args.std_range)
                norm_weight = norm_weight / (2 * self.args.std_range) + 0.5
                num_byte = 2**num_bit
                quantized_weight = torch.round(norm_weight * (num_byte - 1))
                if self.is_entropy:
                    all_param = torch.cat([all_param.int(), quantized_weight.to("cpu").int().view(-1)])
                dequantized_norm_weight = quantized_weight / (num_byte - 1)
                dequantized_norm_weight = (dequantized_norm_weight - 0.5) * 2 * self.args.std_range
                dequantized_weight = dequantized_norm_weight * self.std_wo[i] + self.mean_wo[i]
                layer[0].weight = Parameter(dequantized_weight)
            if self.is_entropy:
                train_dist = self.compute_frequency_distribution(all_param, num_bit)
                num_bit_entropy = self.cross_entropy(train_dist, train_dist)
                return net, num_bit_entropy
            return net, num_bit
            
    def cross_entropy(self, nums1, nums2):
        return -(nums1 * torch.log2(nums2)).sum().item()
    
    def compute_frequency_distribution(self, quantized_layer, num_bit, min_count=1):
        counts = torch.bincount(quantized_layer.view(-1))
        counts = torch.cat([counts, torch.zeros(2**num_bit-len(counts), dtype=counts.dtype).to(counts.device)])
        if min_count:
            counts[counts == 0] += min_count
        frequencies = counts.float() / counts.sum()
        return frequencies

    def get_mean_and_std(self, net):
        if self.args.mode in ["coin", "sw"]:
            mean_wh, mean_bh, std_wh, std_bh = [], [], [], []
            for layer in net.net:
                mean_wh.append(layer[0].weight.mean().item())
                mean_bh.append(layer[0].bias.mean().item())
                std_wh.append(layer[0].weight.std().item())
                std_bh.append(layer[0].bias.std().item())
            self.mean_wh = mean_wh
            self.mean_bh = mean_bh
            self.std_wh = std_wh
            self.std_bh = std_bh
        elif self.args.mode in ["sd", "swd"]:
            mean_wh, mean_bh, std_wh, std_bh = [], [], [], []
            for layer in net.hid_net:
                mean_wh.append(layer[0].weight.mean().item())
                mean_bh.append(layer[0].bias.mean().item())
                std_wh.append(layer[0].weight.std().item())
                std_bh.append(layer[0].bias.std().item())
            mean_wo, mean_bo, std_wo, std_bo = [], [], [], []
            for layer in net.out_net:
                mean_wo.append(layer[0].weight.mean().item())
                mean_bo.append(layer[0].bias.mean().item())
                std_wo.append(layer[0].weight.std().item())
                std_bo.append(layer[0].bias.std().item())
            self.mean_wh = mean_wh
            self.mean_bh = mean_bh
            self.std_wh = std_wh
            self.std_bh = std_bh
            self.mean_wo = mean_wo
            self.mean_bo = mean_bo
            self.std_wo = std_wo
            self.std_bo = std_bo
        

        
        