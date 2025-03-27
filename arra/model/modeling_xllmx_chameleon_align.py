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

from lumina_mgpt.model.adaptor import MlpProjector

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



    def forward(self, input_ids=None, labels=None, image_path=None, training=True, **kwargs):

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


            def load_feature_for_image(image_paths, base_feature_dir=""):
                """
                根据图像路径加载对应的特征
                :param image_path: 图像文件路径
                :param feature_dir: 存放特征文件的文件夹路径
                :return: 图像的特征张量
                """
                features = []
                for image_path in image_paths:
                    sub_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))  # 提取 'p10' 等子目录部分并添加前缀
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]  # 获取图像文件名（不带扩展名）
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "visual_features", f"{image_filename}_visual_feature.pt") # 构造特征文件路径
                    feature = torch.load(feature_path).to("cuda")
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)
                # L2 归一化：将特征向量转换为单位向量
                # batch_features = F.normalize(batch_features, p=2, dim=-1)
                return batch_features

            def load_sam_feature_for_image(image_paths,
                                           base_feature_dir=""):

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
                                           base_feature_dir=""):

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

            def load_feature_for_image_coco(image_paths, base_feature_dir=""):

                features = []
                for image_path in image_paths:
                    # sub_dir = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, "clip_features_L_336", f"{image_filename}_text_feature.pt")
                    feature = torch.load(feature_path).to("cuda")

                    # print("feature", feature.shape)

                    features.append(feature)

                batch_features = torch.cat(features, dim=0)

                # batch_features = (batch_features - batch_features.mean(dim=0, keepdim=True)) / batch_features.std(dim=0, keepdim=True)


                batch_features = F.normalize(batch_features, p=2, dim=-1)

                return batch_features

            def load_feature_for_layer(image_paths, layer, base_feature_dir=""):

                features = []
                for image_path in image_paths:
                    sub_dir = "3con_special_" + os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))
                    sub_dir = sub_dir.replace("p1", "1")
                    image_filename = os.path.splitext(os.path.basename(image_path))[0]  # 获取图像文件名（不带扩展名）
                    # print(sub_dir)
                    feature_path = os.path.join(base_feature_dir, sub_dir, "visual_avgpool_3711", f"layer{layer}", f"{image_filename}.pt")
                    feature = torch.load(feature_path).to("cuda").unsqueeze(dim=0)
                    features.append(feature)
                batch_features = torch.cat(features, dim=0)
                return batch_features

            clip_feature = load_feature_for_image(image_path, "")


            layer_idx = 30

            llm_feature = hidden_states[layer_idx][:, -1, :]
            llm_feature = self.adaptor(llm_feature)  # batch, token_len, embedding_dim   [B,50,4096] -- [B,4096] -- [B,512]



            def cosine_similarity(clip_features, projected_token_features):



                clip_features = F.normalize(clip_features, dim=-1)
                projected_token_features = F.normalize(projected_token_features, dim=-1)


                cosine_sim = torch.sum(clip_features * projected_token_features, dim=-1)

                loss = 1 - cosine_sim

                mean_loss = loss.mean()

                return mean_loss


            cosine_similarity = cosine_similarity(clip_feature, llm_feature)
            additional_loss_dict["similar_loss"] = (cosine_similarity, 1)


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
