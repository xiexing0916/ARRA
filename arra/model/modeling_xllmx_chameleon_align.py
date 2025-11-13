import functools
import logging
import math
from typing import List
import os
import torch
from torch import nn
import torch.nn.functional as F
from .chameleon.modeling_chameleon_align import ChameleonForConditionalGeneration
from .configuration_xllmx_chameleon import ChameleonXLLMXConfig

from arra.model.adaptor import MlpProjector

logger = logging.getLogger(__name__)

default_linear_init = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


__all__ = ["ChameleonXLLMXForConditionalGeneration"]

class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

class ChameleonXLLMXForConditionalGeneration(ChameleonForConditionalGeneration):
    config_class = ChameleonXLLMXConfig

    def __init__(self, config):
        super().__init__(config)
        cfg = AttrDict({
            "input_dim": 4096,
            "n_embed": 512,
            "depth": 2,
            "projector_type": "mlp_gelu",
        })
        self.adaptor = MlpProjector(cfg)
        # self.maxpooling = nn.AdaptiveAvgPool1d(512)
        # self.sim_loss_fn = CosineSimilarityLossWithWeights(3)



    def forward(self, input_ids=None, labels=None, image_path=None, visual_feature_base_dir=None, training=True, **kwargs):

        max_tokens = max([len(_) for _ in input_ids])
        max_tokens = min(max_tokens, self.config.max_position_embeddings)
        input_ids = [_[:max_tokens] for _ in input_ids]
        labels = [_[:max_tokens] for _ in labels]

        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)

        labels = [label + [-100] * (max_tokens - len(label)) for label in labels]
        labels = torch.tensor(labels, dtype=torch.int64, device=self.device)

        # explicit use_cache=False for the following
        # https://github.com/Lightning-AI/pytorch-lightning/issues/19267
        result = ChameleonForConditionalGeneration.forward(
            self, input_ids=input_ids, labels=labels, use_cache=False, output_hidden_states=True, **kwargs
        )

        c_loss = result[0]
        hidden_states = result[2]
        # print(len(hidden_states))
        # print(hidden_states[-1].shape)

        additional_loss_dict = {}
        if self.config.z_loss_weight > 0:
            logits: torch.Tensor = result[1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels >= 0
            z_loss = torch.logsumexp(shift_logits, dim=-1).pow(2)[valid_mask].mean()
            additional_loss_dict["z_loss"] = (z_loss, self.config.z_loss_weight)

            def get_batch_image_features(image_paths, feature_folder, layer_names=None):
                """
                根据一批图像路径和存储特征的文件夹，返回 [layer_num, Batch, Dim] 的特征列表。

                :param image_paths: list, 图像路径列表
                :param feature_folder: str, 存储特征的文件夹路径，
                                       每一层的特征存储在子文件夹中（如 layer4, layer8, layer12）
                :param layer_names: list, 要加载的层的名称列表，默认 ["layer3", "layer7", "layer11"]
                :return: list, 每层的特征 [layer_num, Batch, Dim] 的形式。
                """
                if layer_names is None:
                    layer_names = ["layer3", "layer7", "layer11"]

                layer_features = {layer_name: [] for layer_name in layer_names}

                for image_path in image_paths:
                    try:

                        sub_dir = "3con_special_" + os.path.basename(
                            os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
                        )

                        image_name = os.path.splitext(os.path.basename(image_path))[0] + ".pt"

                        for layer_name in layer_names:
                            layer_folder = os.path.join(feature_folder, sub_dir, "visual_features_3711", layer_name)
                            feature_path = os.path.join(layer_folder, image_name)

                            if not os.path.exists(feature_path):
                                raise FileNotFoundError(f"Feature file not found: {feature_path}")


                            feature = torch.load(feature_path).to("cuda")
                            layer_features[layer_name].append(feature)

                    except FileNotFoundError as e:
                        print(f"Warning: {e}")

                features_list = [torch.stack(layer_features[layer_name], dim=0) for layer_name in layer_names]

                return features_list
            def load_feature_for_image(image_paths, base_feature_dir="./pre_tokenization/cxr"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "visual_features", f"{image_filename}_visual_feature.pt")
                    feature = torch.load(feature_path).to("cuda")
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)
                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features

            def load_feature_for_laion_coco_image(image_paths, base_feature_dir="./pre_tokenization/cxr"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(os.path.dirname(image_path))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, f"{image_filename}_visual_feature.pt")
                    feature = torch.load(feature_path).to("cuda").unsqueeze(dim=0)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)

                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features


            def load_feature_for_blip_image(image_paths, base_feature_dir="./pre_tokenization/cxr"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(os.path.dirname(image_path))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, f"{image_filename}_visual_feature.pt")
                    feature = torch.load(feature_path).to("cuda").unsqueeze(dim=0)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)

                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features

            def load_sam_feature_for_image(image_paths,
                                           base_feature_dir="./pre_tokenization/cxr"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(
                        os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "sam_features",
                                                f"{image_filename}_visual_feature.pt")
                    feature = torch.load(feature_path).to("cuda")
                    # print(feature.shape)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)

                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features

            def load_CLIPL_feature_for_image(image_paths,
                                           base_feature_dir="./pre_tokenization/cxr"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(
                        os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "CLIPL_features",
                                                f"{image_filename}_CLIPL_feature.pt")
                    feature = torch.load(feature_path).to("cuda")
                    # print(feature.shape)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)

                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features

            def load_feature_for_image_coco(image_paths, base_feature_dir="./pre_tokenization/COCO_caption"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, "clip_features_L_336", f"{image_filename}_text_feature.pt")
                    feature = torch.load(feature_path).to("cuda")


                    features.append(feature)

                batch_features = torch.cat(features, dim=0)

                batch_features = F.normalize(batch_features, p=2, dim=-1)

                return batch_features


            def load_feature_for_layer(image_paths, layer, base_feature_dir="./pre_tokenization/mimic_impression"):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = "3con_special_" + os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    sub_dir = sub_dir.replace("p1", "1")
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "visual_avgpool_3711", f"layer{layer}", f"{image_filename}.pt")
                    feature = torch.load(feature_path).to("cuda").unsqueeze(dim=0)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)
                return batch_features

            clip_feature = load_feature_for_image(image_path, visual_feature_base_dir)


            # hidden_states_list = [1, 15, -1]
            # hidden_states_feature_list = []
            # for hidden in hidden_states_list:
            #     llm_feature = self.adaptor(hidden_states[hidden][:, -1, :])   # [B, 512]
            #     hidden_states_feature_list.append(llm_feature)   # [3, B, 512]

            token_id = 8799
            layer_idx = 1
            HYBNEXT_token_strat = -1025
            HYBNEXT_token_end = -1
            token_position = (input_ids == token_id).nonzero(as_tuple=True)[1]
            # print("token_position", token_position)
            # if token_position is not None:
            #     llm_feature = hidden_states[layer_idx][:, -1, :]
            #     llm_feature = self.adaptor(llm_feature)  # batch, token_len, embedding_dim   [B,50,4096] -- [B,4096] -- [B,512]
            # else:
            #     llm_feature = 0
            llm_feature = hidden_states[layer_idx][:, HYBNEXT_token_strat:HYBNEXT_token_end, :]
            llm_feature = self.adaptor(llm_feature)  # batch, token_len, embedding_dim   [B,50,4096] -- [B,4096] -- [B,512]

            # llm_feature = (llm_feature - llm_feature.mean(dim=0, keepdim=True)) / llm_feature.std(dim=0, keepdim=True)
            #
            # llm_feature = F.normalize(llm_feature, p=2, dim=-1)

            # llm_feature = hidden_states[layer_idx][:, token_position, :]  # batch, token_len, embedding_dim   [B,50,4096] -- [B,4096] -- [B,512]
            # llm_feature = self.maxpooling(hidden_states[layer_idx][:, token_position, :])
            # print("llm_feature.shape", llm_feature.shape)

            def cosine_similarity(clip_features, projected_token_features):
                """
                计算视觉特征与自回归模型特定token隐藏层特征的余弦相似性
                :param clip_features: [batch_size, feature_dim] CLIP提取的图像特征
                :param projected_token_features: [batch_size, feature_dim] 自回归模型中特定token的隐藏层经过线性映射后的特征
                :return: 每个样本之间的余弦相似度 [batch_size]
                """
                clip_features = F.normalize(clip_features, dim=-1)
                projected_token_features = F.normalize(projected_token_features, dim=-1)

                cosine_sim = torch.sum(clip_features * projected_token_features, dim=-1)

                loss = 1 - cosine_sim

                mean_loss = loss.mean()

                return mean_loss

            def cosine_similarity_per_token(clip_features, projected_token_features):
                """
                clip_features: [batch, feature_dim]
                projected_token_features: [batch, seq_len, feature_dim]
                return: 标量 loss
                """
                clip_features = F.normalize(clip_features, dim=-1)  # [b, d]
                projected_token_features = F.normalize(projected_token_features, dim=-1)  # [b, l, d]

                clip_features = clip_features.unsqueeze(1)  # [b, 1, d]

                cosine_sim = torch.sum(clip_features * projected_token_features, dim=-1)  # [b, l]

                cosine_sim_mean = cosine_sim.mean(dim=1)  # [b]

                loss = 1 - cosine_sim_mean

                mean_loss = loss.mean()

                return mean_loss

            def cosine_similarity_list_loss(features_list1, features_list2):
                """
                计算两组特征列表之间逐元素的余弦相似度损失。

                :param features_list1: list of tensors, 第一组特征列表
                :param features_list2: list of tensors, 第二组特征列表
                :return: float, 平均余弦相似度损失
                """
                if len(features_list1) != len(features_list2):
                    raise ValueError("Feature lists must have the same length.")

                loss = 0.0
                for f1, f2 in zip(features_list1, features_list2):
                    loss += 1 - F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).mean()

                return loss / len(features_list1)

            # print("llm_list:", hidden_states_feature_list[0].shape)
            # print("clip_feature_list:", clip_feature_list[0].shape)
            # cosine_similarity = self.sim_loss_fn(clip_feature_list, hidden_states_feature_list[::-1])   # [3,6,768]
            # if token_position is not None:
            cosine_similarity = cosine_similarity_per_token(clip_feature, llm_feature)
            additional_loss_dict["similar_loss"] = (cosine_similarity, 1)
            # print("c_loss", c_loss)
            # print("similar_loss", cosine_similarity)

        return c_loss, additional_loss_dict

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.layers), self.lm_head, self.model.embed_tokens]
        if hasattr(self.model, "vqmodel"):  # may be deleted
            modules.append(self.model.vqmodel)
        return modules
    def get_checkpointing_wrap_module_list(self) -> List:
        modules = [
            *list(self.model.layers),
        ]
        return modules
