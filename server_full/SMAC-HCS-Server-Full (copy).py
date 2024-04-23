# clean imports
from __future__ import annotations
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
sys.path.append(os.getcwd())
import torch
from torch import nn
import torch.nn as nn
from PIL import Image
import autogluon.core as ag
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade as ACF
from transformers import FlavaProcessor, FlavaForPreTraining
import os
from collections import Counter
from transformers import AutoTokenizer, Data2VecTextModel
from smac.initial_design.random_design import RandomInitialDesign
from smac.scenario import Scenario
from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.runhistory import RunHistory, StatusType
from ConfigSpace import ConfigurationSpace, UniformIntegerHyperparameter, UniformFloatHyperparameter, \
    CategoricalHyperparameter
# from valse import VALSE
from ConfigSpace import Configuration
from collections import Counter
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, AndConjunction, OrConjunction, InCondition, \
    EqualsCondition, ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
from smac.model.random_forest.random_forest import RandomForest
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, FlavaModel
from ConfigSpace import Configuration
from smac.initial_design.abstract_initial_design import AbstractInitialDesign
# Dummy meta dataset
import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ruamel import yaml
from transformers import FlavaProcessor, FlavaForPreTraining
import os
import json
import torch
from lavis.models import load_model_and_preprocess
from tqdm import tqdm
from PIL import Image
from lavis.models import load_model_and_preprocess
# Creating the dataset loader for the model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import r2_score
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from smac.runhistory.runhistory import RunHistory, InstanceSeedKey, StatusType
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from ConfigSpace import AndConjunction
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer, )
# Creating the dataset loader for the model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import math
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
import random

seed = np.random.randint(10, 10 ** 6)
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMModel
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
# compare standalone models for binary classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from huggingface_hub import login
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchmultimodal.models.albef.image_encoder import ALBEFVisionEncoder
from torchmultimodal.models.albef.model import ALBEFModel, ALBEFModelWithSimilarity
from torchmultimodal.models.albef.multimodal_encoder import ALBEFMultimodalEncoder
from torchmultimodal.modules.encoders.bert_text_encoder import bert_text_encoder
from torchmultimodal.modules.layers.text_embedding import BERTTextEmbeddings
from torchmultimodal.modules.losses.albef import (
    CausalLanguageModelingLoss,
    ImageTextContrastiveLoss,
)
from torchmultimodal.utils.attention import get_causal_attention_mask
from torchmultimodal.utils.common import momentum_update, remove_grad
import re
from typing import List, Tuple, Union

import torch

from torchtext.transforms import PadTransform, Sequential, ToTensor, Truncate
from torchvision import transforms
from transformers.models.bert.tokenization_bert import BertTokenizer
###########################ALBEF PRE-RECS#########################################################################################

_ALBEF_PRETRAINED_URLS = {
    "vqa": "https://download.pytorch.org/models/multimodal/albef/pretrained_vqa_checkpoint.pt",
    "retrieval": "https://download.pytorch.org/models/multimodal/albef/pretrained_retrieval_checkpoint.pt",
}


class PredictionHead(nn.Module):
    """
    Predict the following token autoregressively.

    Args:
        vocab_size (int): The number of different tokens the prediction_head can predict.
        hidden_size (int): The hidden size of the prediction_head.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head.

    Inputs:
        hidden_states (Tensor): The hidden states of preceding tokens.

    Returns:
        Tensor: Prediction scores for the following token.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ALBEFDecoder(nn.Module):
    """
    Generate the prediction scores for answers from image and question hidden states.

    Args:
        text_embeddings (ALBEFTextEmbeddings): Instantiated ALBEFTextEmbeddings.
        multimodal_encoder (ALBEFMultimodalEncoder): Instantiated ALBEFMultimodalEncoder.
        prediction_head (PredictionHead): Instantiated PredictionHead.

    Inputs:
        input_ids (Tensor of shape (batch_size, seq_len)):
            Input ids for input text tokens.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Tensor of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.

    Returns:
        Tensor: Prediction scores for answers.
    """

    def __init__(
        self,
        text_embeddings: BERTTextEmbeddings,
        multimodal_encoder: ALBEFMultimodalEncoder,
        prediction_head: PredictionHead,
    ) -> None:
        super().__init__()
        self.text_embeddings = text_embeddings
        self.multimodal_encoder = multimodal_encoder
        self.prediction_head = prediction_head

    def get_extended_attention_mask_for_decoder(self, attention_mask: Tensor) -> Tensor:
        """
        Apply a causal mask in addition to the padding mask and make the mask broadcastable,
        such that future and masked tokens are ignored.

        Args:
            attention_mask (Tensor):
                Padding mask with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            extended_attention_mask (Tensor):
                The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
        """
        device = attention_mask.device
        batch_size, seq_length = attention_mask.shape
        causal_mask = get_causal_attention_mask(seq_length).to(device)
        causal_mask = causal_mask.repeat(batch_size, 1).view(
            batch_size, seq_length, seq_length
        )
        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        return extended_attention_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.text_embeddings(input_ids)
        attention_mask = self.get_extended_attention_mask_for_decoder(attention_mask)
        decoder_output = self.multimodal_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        prediction_scores = self.prediction_head(decoder_output)
        return prediction_scores


class ALBEFModelForVQA(nn.Module):
    """
    ALBEF Model for VQA finetuning and inference.

    Args:
        model (ALBEFModel): Instantiated ALBEFModel.
        answer_decoder (ALBEFDecoder): Instantiated ALBEFDecoder.
        loss (CausalLanguageModelingLoss): Instantiated CausalLanguageModelingLoss.

    Inputs:
        image (Tensor of shape (B, C, H, W)): Image features.
        question (Tensor of shape (B, L)): Question text features.
        question_atts (Tensor of shape (B, L)): Question attention mask.
        answers (Tensor of shape (N, M)): Answer text features.
        answers_atts (Tensor of shape (N, M)): Answer attention mask.
        ans_weights (Optional[Tensor] of shape (N)): Weights for each answer.
            Required if is_train is True.
        ans_lengths (Optional[List[int]] of length B): Number of answers for each question.
            ans_lengths should sum to N.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Required if is_train is True.
        k (Optional[int]): The number of answers to return for inference.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.

    Returns:
        is_train is True:
            Tensor: The masked language modeling loss for input.
        is_train is False:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
    """

    def __init__(
        self,
        model: ALBEFModel,
        answer_decoder: ALBEFDecoder,
        loss: CausalLanguageModelingLoss,
    ) -> None:
        super().__init__()
        self.model = model
        self.answer_decoder = answer_decoder
        self.loss = loss
        self.answer_decoder_m = copy.deepcopy(self.answer_decoder)
        remove_grad(
            self.answer_decoder_m
        )  # remove gradient for the momentum decoder model

    def _train_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Tensor,
        ans_lengths: List[int],
        alpha: float,
    ) -> Tensor:
        """
        Forward step for training. Encode the inputs with the ALBEFModel.
        Generate pseudo-targets using answer_decoder_m (momentum decoder model).
        Generate answer predictions using answer_decoder.
        Compute masked language modeling loss of the predictions using answers as labels,
            pseudo-targets as soft-labels, and alpha as their interpolation value.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answers_atts (Tensor of shape (N, M)): Answer attention mask.
            ans_weights (Tensor of shape (N)): Weights for each answer.
            ans_lengths (List[int] of length B): Number of answers for each question.
                ans_lengths should sum to N.
            alpha (float): The interpolation value between clm_loss and loss_distill.

        Returns:
            Tensor: The masked language modeling loss for input.
        """
        # get image-question embeddings from the ALBEFModel and format it to match the ans_lengths
        encoder_outputs = self.model(image, question, question_atts)
        (
            encoder_hidden_states,
            encoder_hidden_states_m,
            encoder_attention_mask,
        ) = self._encoder_hidden_states(
            encoder_outputs.multimodal_embeddings,
            encoder_outputs.multimodal_embeddings_m,
            question_atts,
            ans_lengths,
        )

        # use the momentum model to generate pseudo-targets
        with torch.no_grad():
            momentum_update(
                self.answer_decoder, self.answer_decoder_m, self.model.momentum
            )
            prediction_scores_m = self.answer_decoder_m(
                input_ids=answers,
                attention_mask=answers_atts,
                encoder_hidden_states=encoder_hidden_states_m,
                encoder_attention_mask=encoder_attention_mask,
            )

        # generate answer predictions
        prediction_scores = self.answer_decoder(
            input_ids=answers,
            attention_mask=answers_atts,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # compute masked language modeling loss from the prediction scores
        labels = answers.masked_fill(answers == 0, self.loss.mask_token_id)
        loss = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        loss = ans_weights * loss
        loss = loss.sum() / image.size(0)
        return loss

    def _eval_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answer_atts: Tensor,
        k: int = 128,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward step for evaluation. Encode the inputs with the ALBEFModel.
        Generate answer autoregressively using the decoder, starting with the [CLS] token.
        Compute the answer ids and their perspective probabilities of the top k predictions.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answer_atts (Tensor of shape (N, M)): Answer attention mask.
            k (int): The number of answers to return for inference.

        Returns:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
        """
        # get multimodal embeddings from the ALBEFModel and
        # feed it to the decoder as cross attention
        encoder_outputs = self.model(image, question, question_atts)

        # use cls token as the decoder's initial input token
        num_ques = question.size(0)
        start_ids = answers[0, 0].repeat(num_ques, 1)
        atts = torch.ones(start_ids.shape).to(image.device)

        # auto-regressively generates the answer
        prediction_scores = self.answer_decoder(
            input_ids=start_ids,
            attention_mask=atts,
            encoder_hidden_states=encoder_outputs.multimodal_embeddings,
            encoder_attention_mask=question_atts,
        )

        logits = prediction_scores[:, 0, :]
        answer_first_token = answers[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for topk_id in topk_ids:
            input_ids.append(answers.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids)
        input_atts = torch.cat(input_atts)
        targets_ids = input_ids.masked_fill(input_ids == 0, self.loss.mask_token_id)

        question_states = encoder_outputs.multimodal_embeddings.repeat_interleave(
            k, dim=0
        )
        question_atts = question_atts.repeat_interleave(k, dim=0)

        prediction_scores = self.answer_decoder(
            input_ids=input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
        )

        answer_loss = self.loss(targets_ids, prediction_scores)
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    def _encoder_hidden_states(
        self,
        multimodal_embeds: Tensor,
        multimodal_embeds_m: Tensor,
        question_atts: Tensor,
        ans_lengths: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Repeat each image-question input, repeat its embedding and mask to match the number of answers it has.

        Args:
            multimodal_embeds (Tensor): Image-question embeddings.
            multimodal_embeds_m (Tensor): Image-question embeddings from the momentum model.
            question_atts (Tensor): Question attention mask.
            ans_lengths (List[int]): The number of answers each image-question input has.

        Returns:
            encoder_hidden_states (Tensor): Image-question embeddings after the repetition.
            encoder_hidden_states_m (Tensor): Image-question embeddings from the momentum model after the repetition.
            encoder_attention_mask (Tensor): Question attention mask after the repetition.
        """
        encoder_hidden_states = []
        encoder_attention_mask = []
        for b, n in enumerate(ans_lengths):
            encoder_hidden_states += [multimodal_embeds[b]] * n
            encoder_attention_mask += [question_atts[b]] * n
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        encoder_attention_mask = torch.stack(encoder_attention_mask)

        with torch.no_grad():
            encoder_hidden_states_m = []
            for b, n in enumerate(ans_lengths):
                encoder_hidden_states_m += [multimodal_embeds_m[b]] * n
            encoder_hidden_states_m = torch.stack(encoder_hidden_states_m)

        return encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask

    def forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Optional[Tensor] = None,
        ans_lengths: Optional[List[int]] = None,
        alpha: Optional[float] = 0.0,
        k: Optional[int] = 128,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                ans_weights,
                ans_lengths,
                alpha,
            )
        else:
            return self._eval_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                k,
            )


class ALBEFModelForRetrieval(nn.Module):
    """
    ALBEF Model for Retrieval finetuning and inference.
    In training mode, the forward step computes image-text contrastive loss and
    image-text matching loss.
    In evaluation mode, the forward step takes 3 types of input:
        image: encode image input, project and normalize the embeddings.
        text: encode text input, project and normalize the embeddings.
        multimodal: create multimodal embeddings from image and text
            embeddings, and compute image-text matching scores.

    Args:
        model_with_similarity (ALBEFModelWithSimilarity): Instantiated ALBEFModelWithSimilarity.
        itc_loss (ImageTextContrastiveLoss): Instantiated ImageTextContrastiveLoss.
        hidden_size (int): Dimensionality of encoder outputs.

    Inputs:
        image (Optional[Tensor] of shape (B, C, H, W)): Image features.
            Required if is_train is True.
            Required if input_type is "image" or "multimodal".
        text (Optional[Tensor] of shape (B, L)): Text features.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        text_atts (Tensor of shape (B, L)): Text attention mask.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        idx (Tensor of shape (B)): Identifier for each image sample.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Default is 0.
        input_type (Optional[str]): "image", "text", or "multimodal" indicating the encoding type.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.
            Default is True.

    Returns:
        is_train is True:
            Tensor: The sum of itc loss and itm loss.
        is_train is False:
            input_type is "image":
                Tuple[Tensor, Tensor]: Image embeddings and projected image features.
            input_type is "text":
                Tuple[Tensor, Tensor]: Text embeddings and projected text features.
            input_type is "multimodal"
                Tensor: Scores for the retrieval task.
    """

    def __init__(
        self,
        model_with_similarity: ALBEFModelWithSimilarity,
        itc_loss: ImageTextContrastiveLoss,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.model_with_similarity = model_with_similarity
        self.itc_loss = itc_loss
        self.itm_head = nn.Linear(hidden_size, 2)

    def _train_forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
        idx: Tensor,
        alpha: float,
    ) -> Tensor:
        encoder_output = self.model_with_similarity(image, text, text_atts, idx)

        # compute image-text contrastive loss
        similarity_outputs = encoder_output.similarity
        similarity_targets = encoder_output.sim_targets
        itc_loss = self.itc_loss(
            similarity_outputs.sim_i2t,
            similarity_outputs.sim_t2i,
            similarity_outputs.sim_i2t_m,
            similarity_outputs.sim_t2i_m,
            similarity_targets,
            alpha,
        )

        # compute image-text matching loss
        pos_embeddings = encoder_output.multimodal_embeddings[:, 0, :]
        neg_embeddings = encoder_output.multimodal_embeddings_neg[:, 0, :]
        vl_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat(
            [
                torch.ones(pos_embeddings.size(0), dtype=torch.long),
                torch.zeros(neg_embeddings.size(0), dtype=torch.long),
            ],
            dim=0,
        ).to(vl_embeddings.device)
        itm_loss = F.cross_entropy(vl_output, itm_labels)

        loss = itc_loss + itm_loss
        return loss

    def _encode_image(
        self,
        image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_embed = self.model_with_similarity.albef_model.vision_encoder(image)
        image_feat = F.normalize(
            self.model_with_similarity.vision_proj(image_embed[:, 0, :]), dim=-1
        )
        return image_embed, image_feat

    def _encode_text(
        self,
        text: Tensor,
        text_atts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        text_embed = self.model_with_similarity.albef_model.text_encoder(
            text, text_atts
        ).last_hidden_state
        text_feat = F.normalize(
            self.model_with_similarity.text_proj(text_embed[:, 0, :]), dim=-1
        )
        return text_embed, text_feat

    def _image_text_matching_score(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
    ) -> Tensor:
        multimodal_embeds = self.model_with_similarity.albef_model.multimodal_encoder(
            text,
            text_atts,
            image,
        )
        score = self.itm_head(multimodal_embeds[:, 0, :])[:, 1]
        return score

    def _eval_forward(
        self,
        input_type: str,
        image: Optional[Tensor],
        text: Optional[Tensor],
        text_atts: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_type == "image":
            assert image is not None, "image input tensor cannot be None"
            return self._encode_image(image)

        elif input_type == "text":
            assert (
                text is not None and text_atts is not None
            ), "text and text attention mask cannot be None"
            return self._encode_text(text, text_atts)

        elif input_type == "multimodal":
            assert (
                image is not None and text is not None and text_atts is not None
            ), "image embeddings, text embeddings, and text attention mask cannot be None"
            return self._image_text_matching_score(image, text, text_atts)

        else:
            raise ValueError("input_type must be image, text, or multimodal")

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_atts: Optional[Tensor] = None,
        idx: Optional[Tensor] = None,
        alpha: Optional[Tensor] = 0.0,
        input_type: Optional[str] = None,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                text,
                text_atts,
                idx,
                alpha,
            )
        else:
            return self._eval_forward(
                input_type,
                image,
                text,
                text_atts,
            )

def albef_model_for_vqa(
    config: Dict[str, Any], pretrained: bool = False
) -> ALBEFModelForVQA:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = bert_text_encoder(**config["text_encoder_args"])
    question_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    text_embeddings = BERTTextEmbeddings(**config["text_embeddings_args"])
    answer_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    prediction_head = PredictionHead(**config["prediction_head_args"])
    albef_model = ALBEFModel(vision_encoder, text_encoder, question_multimodal_encoder)
    decoder = ALBEFDecoder(text_embeddings, answer_multimodal_encoder, prediction_head)
    loss = CausalLanguageModelingLoss()
    model = ALBEFModelForVQA(albef_model, decoder, loss)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            _ALBEF_PRETRAINED_URLS["vqa"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)
    return model


def albef_model_for_retrieval(
    config: Dict[str, Any], pretrained: bool = False
) -> ALBEFModelForRetrieval:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = bert_text_encoder(**config["text_encoder_args"])
    multimodal_encoder = ALBEFMultimodalEncoder(**config["multimodal_encoder_args"])
    vision_proj = nn.Linear(**config["projection_args"])
    text_proj = nn.Linear(**config["projection_args"])

    albef_model = ALBEFModel(vision_encoder, text_encoder, multimodal_encoder)
    albef_model_with_sim = ALBEFModelWithSimilarity(
        albef_model, vision_proj, text_proj, **config["similarity_args"]
    )
    itc_loss = ImageTextContrastiveLoss()

    model = ALBEFModelForRetrieval(
        albef_model_with_sim, itc_loss, config["hidden_size"]
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            _ALBEF_PRETRAINED_URLS["retrieval"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)
    return model

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)
class ALBEFTextTransform:
    """
    Remove punctuations and trailing spaces in input text and transform it into
    a Tensor of token ids using BERTTokenizer.

    Args:
        pretrained_tokenizer (str): Pretrained tokenizer to use.
            Default: "bert-base-uncased"
        do_pre_process (bool): Whether to pre-process input text.
            Defaults to True.
        truncate (bool): Whether to truncate input text to max_seq_length.
            Defaults to False.
        pad_to_max_seq_len (bool): Whether to pad the sequence to max_seq_length.
        add_end_token (bool): Whether to add the end-of-sentence token.
            Defaults to True.
        max_seq_len (int): The max sequence length after truncating or padding.
            Defaults to 25.
        cls_token_id (int): Value to represent the start of each text.
            Defaults to 101, Hugging Face's BERT cls token id.
        sep_token_id (int): Value to represent the end of each text.
            Defaults to 102, Hugging Face's BERT sep token id.
        pad_token_id (int): Value with which to pad each text so that all texts are the same length.
            Defaults to 0, Hugging Face's BERT pad token id.

    Inputs:
        text (Union[List[str], str]): Input text to transform.
    """

    def __init__(
        self,
        pretrained_tokenizer: str = "bert-base-uncased",
        do_pre_process: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.add_end_token = add_end_token

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.transform = Sequential(
            Truncate(max_seq_len=max_seq_len) if truncate else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
            if pad_to_max_seq_len
            else torch.nn.Identity(),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        tokens = self.tokenizer(text)["input_ids"]
        if not self.add_end_token and tokens[-1] == self.sep_token_id:
            tokens = tokens[:-1]
        input_ids = self.transform(tokens)

        return input_ids


def training_image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )


def testing_image_transform(
    image_size: int = 384,
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=image_interpolation
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )

##################################################################################################################################


access_token = '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'
login(token=access_token, add_to_git_credential=True)

# Define the objective function for optimization
global modality, user_defined_task, user_defined_target, text_column_list, batch_size
batch_size = 10


def stacked_ensemble_RF_L1():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))

    # Define the level 1 model
    level1 = HistGradientBoostingClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


##Add hyperparameter argumets right now -- not needed the, hyperparameters can be overwritten anytime (using set_params)

def stacked_ensemble_XT_L1_1():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = XGBClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_XT_L1_2():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = XGBClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_LGB_L1_1():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier()))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_LGB_L1_2():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))
    level0.append(('RFClassifier', HistGradientBoostingClassifier()))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_LGB_L1_3():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))
    # level0.append(('XtraTreeClassifier', ExtraTreesClassifier()))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_CAT_L1_1():
    level0 = []
    level0.append(('LGBMClassifier', LGBMClassifier(objective='multiclass', num_classes=5)))
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))

    # Define the level 1 model
    level1 = CatBoostClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_CAT_L1_2():
    level0 = []
    level0.append(('LGBMClassifier', LGBMClassifier(objective='multiclass', num_classes=5)))
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('RFClassifier', HistGradientBoostingClassifier()))

    # Define the level 1 model
    level1 = CatBoostClassifier()
    stacked_model = StackingClassifier(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model

_ALBEF_PRETRAINED_URLS = {
    "vqa": "https://download.pytorch.org/models/multimodal/albef/pretrained_vqa_checkpoint.pt",
    "retrieval": "https://download.pytorch.org/models/multimodal/albef/pretrained_retrieval_checkpoint.pt",
}


class PredictionHead(nn.Module):
    """
    Predict the following token autoregressively.

    Args:
        vocab_size (int): The number of different tokens the prediction_head can predict.
        hidden_size (int): The hidden size of the prediction_head.
        layer_norm_eps (float): The epsilon used by the prediction_head normalization layer.
        transform_act_fn (Callable[[Tensor], Tensor]): The activation function in the prediction_head.

    Inputs:
        hidden_states (Tensor): The hidden states of preceding tokens.

    Returns:
        Tensor: Prediction scores for the following token.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        layer_norm_eps: float = 1e-12,
        transform_act_fn: Callable[[Tensor], Tensor] = nn.functional.gelu,
    ) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = transform_act_fn
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class ALBEFDecoder(nn.Module):
    """
    Generate the prediction scores for answers from image and question hidden states.

    Args:
        text_embeddings (ALBEFTextEmbeddings): Instantiated ALBEFTextEmbeddings.
        multimodal_encoder (ALBEFMultimodalEncoder): Instantiated ALBEFMultimodalEncoder.
        prediction_head (PredictionHead): Instantiated PredictionHead.

    Inputs:
        input_ids (Tensor of shape (batch_size, seq_len)):
            Input ids for input text tokens.
        attention_mask (Tensor of shape (batch_size, seq_len)):
            Input attention mask to avoid performing attention on padding token indices.
        encoder_hidden_states (Tensor of shape (batch_size, encoder_seq_len, hidden_size)):
            The encoder hidden states.
        encoder_attention_mask (Tensor of shape (batch_size, encoder_seq_len)):
            The attention mask for encoder hidden states.

    Returns:
        Tensor: Prediction scores for answers.
    """

    def __init__(
        self,
        text_embeddings: BERTTextEmbeddings,
        multimodal_encoder: ALBEFMultimodalEncoder,
        prediction_head: PredictionHead,
    ) -> None:
        super().__init__()
        self.text_embeddings = text_embeddings
        self.multimodal_encoder = multimodal_encoder
        self.prediction_head = prediction_head

    def get_extended_attention_mask_for_decoder(self, attention_mask: Tensor) -> Tensor:
        """
        Apply a causal mask in addition to the padding mask and make the mask broadcastable,
        such that future and masked tokens are ignored.

        Args:
            attention_mask (Tensor):
                Padding mask with ones indicating tokens to attend to, zeros for tokens to ignore.

        Returns:
            extended_attention_mask (Tensor):
                The broadcastable attention mask, with the same dtype as ``attention_mask.dtype``.
        """
        device = attention_mask.device
        batch_size, seq_length = attention_mask.shape
        causal_mask = get_causal_attention_mask(seq_length).to(device)
        causal_mask = causal_mask.repeat(batch_size, 1).view(
            batch_size, seq_length, seq_length
        )
        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        return extended_attention_mask

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        encoder_attention_mask: Tensor,
    ) -> Tensor:
        hidden_states = self.text_embeddings(input_ids)
        attention_mask = self.get_extended_attention_mask_for_decoder(attention_mask)
        decoder_output = self.multimodal_encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        prediction_scores = self.prediction_head(decoder_output)
        return prediction_scores


class ALBEFModelForVQA(nn.Module):
    """
    ALBEF Model for VQA finetuning and inference.

    Args:
        model (ALBEFModel): Instantiated ALBEFModel.
        answer_decoder (ALBEFDecoder): Instantiated ALBEFDecoder.
        loss (CausalLanguageModelingLoss): Instantiated CausalLanguageModelingLoss.

    Inputs:
        image (Tensor of shape (B, C, H, W)): Image features.
        question (Tensor of shape (B, L)): Question text features.
        question_atts (Tensor of shape (B, L)): Question attention mask.
        answers (Tensor of shape (N, M)): Answer text features.
        answers_atts (Tensor of shape (N, M)): Answer attention mask.
        ans_weights (Optional[Tensor] of shape (N)): Weights for each answer.
            Required if is_train is True.
        ans_lengths (Optional[List[int]] of length B): Number of answers for each question.
            ans_lengths should sum to N.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Required if is_train is True.
        k (Optional[int]): The number of answers to return for inference.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.

    Returns:
        is_train is True:
            Tensor: The masked language modeling loss for input.
        is_train is False:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
    """

    def __init__(
        self,
        model: ALBEFModel,
        answer_decoder: ALBEFDecoder,
        loss: CausalLanguageModelingLoss,
    ) -> None:
        super().__init__()
        self.model = model
        self.answer_decoder = answer_decoder
        self.loss = loss
        self.answer_decoder_m = copy.deepcopy(self.answer_decoder)
        remove_grad(
            self.answer_decoder_m
        )  # remove gradient for the momentum decoder model

    def _train_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Tensor,
        ans_lengths: List[int],
        alpha: float,
    ) -> Tensor:
        """
        Forward step for training. Encode the inputs with the ALBEFModel.
        Generate pseudo-targets using answer_decoder_m (momentum decoder model).
        Generate answer predictions using answer_decoder.
        Compute masked language modeling loss of the predictions using answers as labels,
            pseudo-targets as soft-labels, and alpha as their interpolation value.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answers_atts (Tensor of shape (N, M)): Answer attention mask.
            ans_weights (Tensor of shape (N)): Weights for each answer.
            ans_lengths (List[int] of length B): Number of answers for each question.
                ans_lengths should sum to N.
            alpha (float): The interpolation value between clm_loss and loss_distill.

        Returns:
            Tensor: The masked language modeling loss for input.
        """
        # get image-question embeddings from the ALBEFModel and format it to match the ans_lengths
        encoder_outputs = self.model(image, question, question_atts)
        (
            encoder_hidden_states,
            encoder_hidden_states_m,
            encoder_attention_mask,
        ) = self._encoder_hidden_states(
            encoder_outputs.multimodal_embeddings,
            encoder_outputs.multimodal_embeddings_m,
            question_atts,
            ans_lengths,
        )

        # use the momentum model to generate pseudo-targets
        with torch.no_grad():
            momentum_update(
                self.answer_decoder, self.answer_decoder_m, self.model.momentum
            )
            prediction_scores_m = self.answer_decoder_m(
                input_ids=answers,
                attention_mask=answers_atts,
                encoder_hidden_states=encoder_hidden_states_m,
                encoder_attention_mask=encoder_attention_mask,
            )

        # generate answer predictions
        prediction_scores = self.answer_decoder(
            input_ids=answers,
            attention_mask=answers_atts,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # compute masked language modeling loss from the prediction scores
        labels = answers.masked_fill(answers == 0, self.loss.mask_token_id)
        loss = self.loss(labels, prediction_scores, prediction_scores_m, alpha)
        loss = ans_weights * loss
        loss = loss.sum() / image.size(0)
        return loss

    def _eval_forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answer_atts: Tensor,
        k: int = 128,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward step for evaluation. Encode the inputs with the ALBEFModel.
        Generate answer autoregressively using the decoder, starting with the [CLS] token.
        Compute the answer ids and their perspective probabilities of the top k predictions.

        Inputs:
            image (Tensor of shape (B, C, H, W)): Image features.
            question (Tensor of shape (B, L)): Question text features.
            question_atts (Tensor of shape (B, L)): Question attention mask.
            answers (Tensor of shape (N, M)): Answer text features.
            answer_atts (Tensor of shape (N, M)): Answer attention mask.
            k (int): The number of answers to return for inference.

        Returns:
            Tuple[Tensor, Tensor]: The ids and probabilities for the top k predicted answers.
        """
        # get multimodal embeddings from the ALBEFModel and
        # feed it to the decoder as cross attention
        encoder_outputs = self.model(image, question, question_atts)

        # use cls token as the decoder's initial input token
        num_ques = question.size(0)
        start_ids = answers[0, 0].repeat(num_ques, 1)
        atts = torch.ones(start_ids.shape).to(image.device)

        # auto-regressively generates the answer
        prediction_scores = self.answer_decoder(
            input_ids=start_ids,
            attention_mask=atts,
            encoder_hidden_states=encoder_outputs.multimodal_embeddings,
            encoder_attention_mask=question_atts,
        )

        logits = prediction_scores[:, 0, :]
        answer_first_token = answers[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        input_ids = []
        input_atts = []
        for topk_id in topk_ids:
            input_ids.append(answers.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids)
        input_atts = torch.cat(input_atts)
        targets_ids = input_ids.masked_fill(input_ids == 0, self.loss.mask_token_id)

        question_states = encoder_outputs.multimodal_embeddings.repeat_interleave(
            k, dim=0
        )
        question_atts = question_atts.repeat_interleave(k, dim=0)

        prediction_scores = self.answer_decoder(
            input_ids=input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
        )

        answer_loss = self.loss(targets_ids, prediction_scores)
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)

        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs

    def _encoder_hidden_states(
        self,
        multimodal_embeds: Tensor,
        multimodal_embeds_m: Tensor,
        question_atts: Tensor,
        ans_lengths: List[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Repeat each image-question input, repeat its embedding and mask to match the number of answers it has.

        Args:
            multimodal_embeds (Tensor): Image-question embeddings.
            multimodal_embeds_m (Tensor): Image-question embeddings from the momentum model.
            question_atts (Tensor): Question attention mask.
            ans_lengths (List[int]): The number of answers each image-question input has.

        Returns:
            encoder_hidden_states (Tensor): Image-question embeddings after the repetition.
            encoder_hidden_states_m (Tensor): Image-question embeddings from the momentum model after the repetition.
            encoder_attention_mask (Tensor): Question attention mask after the repetition.
        """
        encoder_hidden_states = []
        encoder_attention_mask = []
        for b, n in enumerate(ans_lengths):
            encoder_hidden_states += [multimodal_embeds[b]] * n
            encoder_attention_mask += [question_atts[b]] * n
        encoder_hidden_states = torch.stack(encoder_hidden_states)
        encoder_attention_mask = torch.stack(encoder_attention_mask)

        with torch.no_grad():
            encoder_hidden_states_m = []
            for b, n in enumerate(ans_lengths):
                encoder_hidden_states_m += [multimodal_embeds_m[b]] * n
            encoder_hidden_states_m = torch.stack(encoder_hidden_states_m)

        return encoder_hidden_states, encoder_hidden_states_m, encoder_attention_mask

    def forward(
        self,
        image: Tensor,
        question: Tensor,
        question_atts: Tensor,
        answers: Tensor,
        answers_atts: Tensor,
        ans_weights: Optional[Tensor] = None,
        ans_lengths: Optional[List[int]] = None,
        alpha: Optional[float] = 0.0,
        k: Optional[int] = 128,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                ans_weights,
                ans_lengths,
                alpha,
            )
        else:
            return self._eval_forward(
                image,
                question,
                question_atts,
                answers,
                answers_atts,
                k,
            )


class ALBEFModelForRetrieval(nn.Module):
    """
    ALBEF Model for Retrieval finetuning and inference.
    In training mode, the forward step computes image-text contrastive loss and
    image-text matching loss.
    In evaluation mode, the forward step takes 3 types of input:
        image: encode image input, project and normalize the embeddings.
        text: encode text input, project and normalize the embeddings.
        multimodal: create multimodal embeddings from image and text
            embeddings, and compute image-text matching scores.

    Args:
        model_with_similarity (ALBEFModelWithSimilarity): Instantiated ALBEFModelWithSimilarity.
        itc_loss (ImageTextContrastiveLoss): Instantiated ImageTextContrastiveLoss.
        hidden_size (int): Dimensionality of encoder outputs.

    Inputs:
        image (Optional[Tensor] of shape (B, C, H, W)): Image features.
            Required if is_train is True.
            Required if input_type is "image" or "multimodal".
        text (Optional[Tensor] of shape (B, L)): Text features.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        text_atts (Tensor of shape (B, L)): Text attention mask.
            Required if is_train is True.
            Required if input_type is "text" or "multimodal".
        idx (Tensor of shape (B)): Identifier for each image sample.
            Required if is_train is True.
        alpha (Optional[float]): The interpolation value between clm_loss and loss_distill.
            Default is 0.
        input_type (Optional[str]): "image", "text", or "multimodal" indicating the encoding type.
            Required if is_train is False.
        is_train (Optional[bool]): Whether the model is in training.
            Default is True.

    Returns:
        is_train is True:
            Tensor: The sum of itc loss and itm loss.
        is_train is False:
            input_type is "image":
                Tuple[Tensor, Tensor]: Image embeddings and projected image features.
            input_type is "text":
                Tuple[Tensor, Tensor]: Text embeddings and projected text features.
            input_type is "multimodal"
                Tensor: Scores for the retrieval task.
    """

    def __init__(
        self,
        model_with_similarity: ALBEFModelWithSimilarity,
        itc_loss: ImageTextContrastiveLoss,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.model_with_similarity = model_with_similarity
        self.itc_loss = itc_loss
        self.itm_head = nn.Linear(hidden_size, 2)

    def _train_forward(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
        idx: Tensor,
        alpha: float,
    ) -> Tensor:
        encoder_output = self.model_with_similarity(image, text, text_atts, idx)

        # compute image-text contrastive loss
        similarity_outputs = encoder_output.similarity
        similarity_targets = encoder_output.sim_targets
        itc_loss = self.itc_loss(
            similarity_outputs.sim_i2t,
            similarity_outputs.sim_t2i,
            similarity_outputs.sim_i2t_m,
            similarity_outputs.sim_t2i_m,
            similarity_targets,
            alpha,
        )

        # compute image-text matching loss
        pos_embeddings = encoder_output.multimodal_embeddings[:, 0, :]
        neg_embeddings = encoder_output.multimodal_embeddings_neg[:, 0, :]
        vl_embeddings = torch.cat([pos_embeddings, neg_embeddings], dim=0)
        vl_output = self.itm_head(vl_embeddings)
        itm_labels = torch.cat(
            [
                torch.ones(pos_embeddings.size(0), dtype=torch.long),
                torch.zeros(neg_embeddings.size(0), dtype=torch.long),
            ],
            dim=0,
        ).to(vl_embeddings.device)
        itm_loss = F.cross_entropy(vl_output, itm_labels)

        loss = itc_loss + itm_loss
        return loss

    def _encode_image(
        self,
        image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        image_embed = self.model_with_similarity.albef_model.vision_encoder(image)
        image_feat = F.normalize(
            self.model_with_similarity.vision_proj(image_embed[:, 0, :]), dim=-1
        )
        return image_embed, image_feat

    def _encode_text(
        self,
        text: Tensor,
        text_atts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        text_embed = self.model_with_similarity.albef_model.text_encoder(
            text, text_atts
        ).last_hidden_state
        text_feat = F.normalize(
            self.model_with_similarity.text_proj(text_embed[:, 0, :]), dim=-1
        )
        return text_embed, text_feat

    def _image_text_matching_score(
        self,
        image: Tensor,
        text: Tensor,
        text_atts: Tensor,
    ) -> Tensor:
        multimodal_embeds = self.model_with_similarity.albef_model.multimodal_encoder(
            text,
            text_atts,
            image,
        )
        score = self.itm_head(multimodal_embeds[:, 0, :])[:, 1]
        return score

    def _eval_forward(
        self,
        input_type: str,
        image: Optional[Tensor],
        text: Optional[Tensor],
        text_atts: Optional[Tensor],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if input_type == "image":
            assert image is not None, "image input tensor cannot be None"
            return self._encode_image(image)

        elif input_type == "text":
            assert (
                text is not None and text_atts is not None
            ), "text and text attention mask cannot be None"
            return self._encode_text(text, text_atts)

        elif input_type == "multimodal":
            assert (
                image is not None and text is not None and text_atts is not None
            ), "image embeddings, text embeddings, and text attention mask cannot be None"
            return self._image_text_matching_score(image, text, text_atts)

        else:
            raise ValueError("input_type must be image, text, or multimodal")

    def forward(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
        text_atts: Optional[Tensor] = None,
        idx: Optional[Tensor] = None,
        alpha: Optional[Tensor] = 0.0,
        input_type: Optional[str] = None,
        is_train: Optional[bool] = True,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if is_train:
            return self._train_forward(
                image,
                text,
                text_atts,
                idx,
                alpha,
            )
        else:
            return self._eval_forward(
                input_type,
                image,
                text,
                text_atts,
            )


def albef_model_for_vqa(
    config: Dict[str, Any], pretrained: bool = False
) -> ALBEFModelForVQA:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = bert_text_encoder(**config["text_encoder_args"])
    question_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    text_embeddings = BERTTextEmbeddings(**config["text_embeddings_args"])
    answer_multimodal_encoder = ALBEFMultimodalEncoder(
        **config["multimodal_encoder_args"]
    )
    prediction_head = PredictionHead(**config["prediction_head_args"])
    albef_model = ALBEFModel(vision_encoder, text_encoder, question_multimodal_encoder)
    decoder = ALBEFDecoder(text_embeddings, answer_multimodal_encoder, prediction_head)
    loss = CausalLanguageModelingLoss()
    model = ALBEFModelForVQA(albef_model, decoder, loss)

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            _ALBEF_PRETRAINED_URLS["vqa"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)
    return model


def albef_model_for_retrieval(
    config: Dict[str, Any], pretrained: bool = False
) -> ALBEFModelForRetrieval:
    vision_encoder = ALBEFVisionEncoder(**config["vision_encoder_args"])
    text_encoder = bert_text_encoder(**config["text_encoder_args"])
    multimodal_encoder = ALBEFMultimodalEncoder(**config["multimodal_encoder_args"])
    vision_proj = nn.Linear(**config["projection_args"])
    text_proj = nn.Linear(**config["projection_args"])

    albef_model = ALBEFModel(vision_encoder, text_encoder, multimodal_encoder)
    albef_model_with_sim = ALBEFModelWithSimilarity(
        albef_model, vision_proj, text_proj, **config["similarity_args"]
    )
    itc_loss = ImageTextContrastiveLoss()

    model = ALBEFModelForRetrieval(
        albef_model_with_sim, itc_loss, config["hidden_size"]
    )

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            _ALBEF_PRETRAINED_URLS["retrieval"], map_location="cpu"
        )
        model.load_state_dict(checkpoint)
    return model

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD_DEV = (0.26862954, 0.26130258, 0.27577711)
class ALBEFTextTransform:
    """
    Remove punctuations and trailing spaces in input text and transform it into
    a Tensor of token ids using BERTTokenizer.

    Args:
        pretrained_tokenizer (str): Pretrained tokenizer to use.
            Default: "bert-base-uncased"
        do_pre_process (bool): Whether to pre-process input text.
            Defaults to True.
        truncate (bool): Whether to truncate input text to max_seq_length.
            Defaults to False.
        pad_to_max_seq_len (bool): Whether to pad the sequence to max_seq_length.
        add_end_token (bool): Whether to add the end-of-sentence token.
            Defaults to True.
        max_seq_len (int): The max sequence length after truncating or padding.
            Defaults to 25.
        cls_token_id (int): Value to represent the start of each text.
            Defaults to 101, Hugging Face's BERT cls token id.
        sep_token_id (int): Value to represent the end of each text.
            Defaults to 102, Hugging Face's BERT sep token id.
        pad_token_id (int): Value with which to pad each text so that all texts are the same length.
            Defaults to 0, Hugging Face's BERT pad token id.

    Inputs:
        text (Union[List[str], str]): Input text to transform.
    """

    def __init__(
        self,
        pretrained_tokenizer: str = "bert-base-uncased",
        do_pre_process: bool = True,
        truncate: bool = False,
        pad_to_max_seq_len: bool = False,
        add_end_token: bool = True,
        max_seq_len: int = 25,
        cls_token_id: int = 101,
        sep_token_id: int = 102,
        pad_token_id: int = 0,
    ):
        self.do_pre_process = do_pre_process
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.add_end_token = add_end_token

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_tokenizer)
        self.transform = Sequential(
            Truncate(max_seq_len=max_seq_len) if truncate else torch.nn.Identity(),
            ToTensor(padding_value=self.pad_token_id),
            PadTransform(max_length=max_seq_len, pad_value=self.pad_token_id)
            if pad_to_max_seq_len
            else torch.nn.Identity(),
        )

    def pre_process(self, text: str) -> str:
        text = (
            re.sub(
                r"([,.'!?\"()*#:;~])",
                "",
                text,
            )
            .replace("-", " ")
            .replace("/", " ")
        )
        text = text.rstrip(" ")

        return text

    def __call__(self, text: Union[List[str], str]) -> torch.Tensor:
        if self.do_pre_process:
            if isinstance(text, str):
                text = self.pre_process(text)
            else:
                text = [self.pre_process(t) for t in text]
        tokens = self.tokenizer(text)["input_ids"]
        if not self.add_end_token and tokens[-1] == self.sep_token_id:
            tokens = tokens[:-1]
        input_ids = self.transform(tokens)

        return input_ids


def training_image_transform(
    image_size: int = 384,
    scale: Tuple[float, float] = (0.5, 1.0),
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=scale, interpolation=image_interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(2, 7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )


def testing_image_transform(
    image_size: int = 384,
    image_interpolation=transforms.InterpolationMode.BICUBIC,
    mean: Tuple[float, float, float] = MEAN,
    std_dev: Tuple[float, float, float] = STD_DEV,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size), interpolation=image_interpolation
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std_dev),
        ]
    )


def weighted_ensemble_RF_L1():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))
    level0.append(('XGBClassifier', XGBClassifier(objective='multiclass')))

    # Define the level 1 model
    level1 = RandomForestClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_XT_L1_1():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = XGBClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_XT_L1_2():
    level0 = []
    level0.append(('LGBClassifier', LGBMClassifier(objective='multiclass', num_class=5)))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = XGBClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_LGB_L1_1():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier()))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_LGB_L1_2():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))
    level0.append(('RFClassifier', HistGradientBoostingClassifier()))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_LGB_L1_3():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('CatBoostClassifier', CatBoostClassifier(objective='MultiClass', classes_count=5)))

    # Define the level 1 model
    level1 = LGBMClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_CAT_L1_1():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('LGBMClassifier', LGBMClassifier(objective='multiclass', num_classes=5)))

    # Define the level 1 model
    level1 = CatBoostClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def weighted_ensemble_CAT_L1_2():
    level0 = []
    level0.append(('XGBoostClassifier', XGBClassifier(objective='multiclass')))
    level0.append(('LGBMClassifier', LGBMClassifier(objective='multiclass', num_classes=5)))
    level0.append(('RFClassifier', HistGradientBoostingClassifier()))

    # Define the level 1 model
    level1 = CatBoostClassifier()
    stacked_model = VotingClassifier(estimators=level0, voting='soft')
    return stacked_model


def stacked_ensemble_r__RF_L1():  ##Input sh
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))
    level0.append(('XGBoostRegressor', XGBRegressor()))

    # Define the level 1 model
    level1 = RandomForestRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


##Add hyperparameter argumets right now -- not needed the, hyperparameters can be overwritten anytime (using set_params)

def stacked_ensemble_r_XT_L1_1():
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = XGBRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_XT_L1_2():
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = XGBRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_LGB_L1_1():
    level0 = []
    level0.append(('CatBoostRegressor', CatBoostRegressor()))
    level0.append(('XGBoostRegressor', XGBRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_LGB_L1_2():
    level0 = []

    level0.append(('CatBoostRegressor', CatBoostRegressor()))
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('RFRegressor', HistGradientBoostingRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_LGB_L1_3():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_CAT_L1_1():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('LGBMRegressor', LGBMRegressor()))

    # Define the level 1 model
    level1 = CatBoostRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def stacked_ensemble_r_CAT_L1_2():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('LGBMRegressor', LGBMRegressor()))
    level0.append(('RFRegressor', HistGradientBoostingRegressor()))

    # Define the level 1 model
    level1 = CatBoostRegressor()
    stacked_model = StackingRegressor(estimators=level0, final_estimator=level1, cv=3)
    return stacked_model


def weighted_ensemble_r__RF_L1():
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))
    level0.append(('XGBoostRegressor', XGBRegressor()))

    # Define the level 1 model
    level1 = RandomForestRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_XT_L1_1():
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = XGBRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_XT_L1_2():
    level0 = []
    level0.append(('LGBRegressor', LGBMRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = XGBRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_LGB_L1_1():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_LGB_L1_2():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))
    level0.append(('RFRegressor', HistGradientBoostingRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_LGB_L1_3():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('CatBoostRegressor', CatBoostRegressor()))

    # Define the level 1 model
    level1 = LGBMRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_CAT_L1_1():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('LGBMRegressor', LGBMRegressor()))

    # Define the level 1 model
    level1 = CatBoostRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


def weighted_ensemble_r_CAT_L1_2():
    level0 = []
    level0.append(('XGBoostRegressor', XGBRegressor()))
    level0.append(('LGBMRegressor', LGBMRegressor()))
    level0.append(('RFRegressor', HistGradientBoostingRegressor()))

    # Define the level 1 model
    level1 = CatBoostRegressor()
    weighted_model = VotingRegressor(estimators=level0)
    return weighted_model


stacked_ensemble_RF_L1 = stacked_ensemble_RF_L1()
stacked_ensemble_XT_L1_1 = stacked_ensemble_XT_L1_1()
stacked_ensemble_XT_L1_2 = stacked_ensemble_XT_L1_2()
stacked_ensemble_LGB_L1_1 = stacked_ensemble_LGB_L1_1()
stacked_ensemble_LGB_L1_2 = stacked_ensemble_LGB_L1_2()
stacked_ensemble_LGB_L1_3 = stacked_ensemble_LGB_L1_3()
stacked_ensemble_CAT_L1_1 = stacked_ensemble_CAT_L1_1()
stacked_ensemble_CAT_L1_2 = stacked_ensemble_CAT_L1_2()

weighted_ensemble_RF_L1 = weighted_ensemble_RF_L1()
weighted_ensemble_XT_L1_1 = weighted_ensemble_XT_L1_1()
weighted_ensemble_XT_L1_2 = weighted_ensemble_XT_L1_2()
weighted_ensemble_LGB_L1_1 = weighted_ensemble_LGB_L1_1()
weighted_ensemble_LGB_L1_2 = weighted_ensemble_LGB_L1_2()
weighted_ensemble_LGB_L1_3 = weighted_ensemble_LGB_L1_3()
weighted_ensemble_CAT_L1_1 = weighted_ensemble_CAT_L1_1()
weighted_ensemble_CAT_L1_2 = weighted_ensemble_CAT_L1_2()

stacked_ensemble_r__RF_L1 = stacked_ensemble_r__RF_L1()
stacked_ensemble_r_XT_L1_1 = stacked_ensemble_r_XT_L1_1()
stacked_ensemble_r_XT_L1_2 = stacked_ensemble_r_XT_L1_2()
stacked_ensemble_r_LGB_L1_1 = stacked_ensemble_r_LGB_L1_1()
stacked_ensemble_r_LGB_L1_2 = stacked_ensemble_r_LGB_L1_2()
stacked_ensemble_r_LGB_L1_3 = stacked_ensemble_r_LGB_L1_3()
stacked_ensemble_r_CAT_L1_1 = stacked_ensemble_r_CAT_L1_1()
stacked_ensemble_r_CAT_L1_2 = stacked_ensemble_r_CAT_L1_2()

weighted_ensemble_r__RF_L1 = weighted_ensemble_r__RF_L1()
weighted_ensemble_r_XT_L1_1 = weighted_ensemble_r_XT_L1_1()
weighted_ensemble_r_XT_L1_2 = weighted_ensemble_r_XT_L1_2()
weighted_ensemble_r_LGB_L1_1 = weighted_ensemble_r_LGB_L1_1()
weighted_ensemble_r_LGB_L1_2 = weighted_ensemble_r_LGB_L1_2()
weighted_ensemble_r_LGB_L1_3 = weighted_ensemble_r_LGB_L1_3()
weighted_ensemble_r_CAT_L1_1 = weighted_ensemble_r_CAT_L1_1()
weighted_ensemble_r_CAT_L1_2 = weighted_ensemble_r_CAT_L1_2()

# In[4]:


downstrem_choices = ['CatBoostClassifier',
                     'CatBoostRegressor',
                     'XGBoostClassifier',
                     'XGBoostRegressor',
                     'LGBMClassifier',
                     'LGBMRegressor',
                     'stacked_ensemble_RF_L1',
                     'stacked_ensemble_XT_L1_1',
                     'stacked_ensemble_XT_L1_2',
                     'stacked_ensemble_LGB_L1_1',
                     'stacked_ensemble_LGB_L1_2',
                     'stacked_ensemble_LGB_L1_3',
                     'stacked_ensemble_CAT_L1_1',
                     'stacked_ensemble_CAT_L1_2',
                     'weighted_ensemble_RF_L1',
                     'weighted_ensemble_XT_L1_1',
                     'weighted_ensemble_XT_L1_2',
                     'weighted_ensemble_LGB_L1_1',
                     'weighted_ensemble_LGB_L1_2',
                     'weighted_ensemble_LGB_L1_3',
                     'weighted_ensemble_CAT_L1_1',
                     'weighted_ensemble_CAT_L1_2',
                     'stacked_ensemble_r__RF_L1',
                     'stacked_ensemble_r_XT_L1_1',
                     'stacked_ensemble_r_XT_L1_2',
                     'stacked_ensemble_r_LGB_L1_1',
                     'stacked_ensemble_r_LGB_L1_2',
                     'stacked_ensemble_r_LGB_L1_3',
                     'stacked_ensemble_r_CAT_L1_1',
                     'stacked_ensemble_r_CAT_L1_2',
                     'weighted_ensemble_r__RF_L1',
                     'weighted_ensemble_r_XT_L1_1',
                     'weighted_ensemble_r_XT_L1_2',
                     'weighted_ensemble_r_LGB_L1_1',
                     'weighted_ensemble_r_LGB_L1_2',
                     'weighted_ensemble_r_LGB_L1_3',
                     'weighted_ensemble_r_CAT_L1_1',
                     'weighted_ensemble_r_CAT_L1_2']

models_consisting_lgb = ['LGBMClassifier',
                         'LGBMRegressor',

                         'stacked_ensemble_RF_L1',
                         'stacked_ensemble_XT_L1_1',
                         'stacked_ensemble_XT_L1_2',
                         'stacked_ensemble_CAT_L1_1',
                         'stacked_ensemble_CAT_L1_2',

                         'weighted_ensemble_RF_L1',
                         'weighted_ensemble_XT_L1_1',
                         'weighted_ensemble_XT_L1_2',
                         'weighted_ensemble_LGB_L1_1',
                         'weighted_ensemble_LGB_L1_2',
                         'weighted_ensemble_LGB_L1_3',
                         'weighted_ensemble_CAT_L1_1',
                         'weighted_ensemble_CAT_L1_2',

                         'stacked_ensemble_r__RF_L1',
                         'stacked_ensemble_r_XT_L1_1',
                         'stacked_ensemble_r_XT_L1_2',
                         'stacked_ensemble_r_LGB_L1_1',
                         'stacked_ensemble_r_LGB_L1_2',
                         'stacked_ensemble_r_LGB_L1_3',
                         'stacked_ensemble_r_CAT_L1_1',
                         'stacked_ensemble_r_CAT_L1_2',
                         'weighted_ensemble_r__RF_L1',
                         'weighted_ensemble_r__RF_L1',
                         'weighted_ensemble_r_XT_L1_2',
                         'weighted_ensemble_r_CAT_L1_1',
                         'weighted_ensemble_r_CAT_L1_2']

models_consisting_knn = ['KNNClassifier',
                         'weighted_ensemble_LGB_L1_1',
                         'weighted_ensemble_XT_L1_2',
                         'weighted_ensemble_XT_L1_1',
                         'stacked_ensemble_LGB_L1_1',
                         'stacked_ensemble_XT_L1_2',
                         'stacked_ensemble_XT_L1_1']

regression_model_list = ['CatBoostRegressor',
                         'XGBoostRegressor',
                         'LGBMRegressor',
                         'stacked_ensemble_r__RF_L1',
                         'stacked_ensemble_r_XT_L1_1',
                         'stacked_ensemble_r_XT_L1_2',
                         'stacked_ensemble_r_LGB_L1_1',
                         'stacked_ensemble_r_LGB_L1_2',
                         'stacked_ensemble_r_LGB_L1_3',
                         'stacked_ensemble_r_CAT_L1_1',
                         'stacked_ensemble_r_CAT_L1_2',
                         'weighted_ensemble_r__RF_L1',
                         'weighted_ensemble_r_XT_L1_1',
                         'weighted_ensemble_r_XT_L1_2',
                         'weighted_ensemble_r_LGB_L1_1',
                         'weighted_ensemble_r_LGB_L1_2',
                         'weighted_ensemble_r_LGB_L1_3',
                         'weighted_ensemble_r_CAT_L1_1',
                         'weighted_ensemble_r_CAT_L1_2']

classification_model_list = ['CatBoostClassifier',
                             'XGBoostClassifier',
                             'LGBMClassifier',
                             'stacked_ensemble_RF_L1',
                             'stacked_ensemble_XT_L1_1',
                             'stacked_ensemble_XT_L1_2',
                             'stacked_ensemble_LGB_L1_1',
                             'stacked_ensemble_LGB_L1_2',
                             'stacked_ensemble_LGB_L1_3',
                             'stacked_ensemble_CAT_L1_1',
                             'stacked_ensemble_CAT_L1_2',
                             'weighted_ensemble_RF_L1',
                             'weighted_ensemble_XT_L1_1',
                             'weighted_ensemble_XT_L1_2',
                             'weighted_ensemble_LGB_L1_1',
                             'weighted_ensemble_LGB_L1_2',
                             'weighted_ensemble_LGB_L1_3',
                             'weighted_ensemble_CAT_L1_1',
                             'weighted_ensemble_CAT_L1_2']

ensemble_clf_models = {'weighted_ensemble_RF_L1': weighted_ensemble_RF_L1,
                       'weighted_ensemble_XT_L1_1': weighted_ensemble_XT_L1_1,
                       'weighted_ensemble_XT_L1_2': weighted_ensemble_XT_L1_2,
                       'weighted_ensemble_LGB_L1_1': weighted_ensemble_LGB_L1_1,
                       'weighted_ensemble_LGB_L1_2': weighted_ensemble_LGB_L1_2,
                       'weighted_ensemble_LGB_L1_3': weighted_ensemble_LGB_L1_3,
                       'weighted_ensemble_CAT_L1_1': weighted_ensemble_CAT_L1_1,
                       'weighted_ensemble_CAT_L1_2': weighted_ensemble_CAT_L1_2,
                       'stacked_ensemble_RF_L1': stacked_ensemble_RF_L1,
                       'stacked_ensemble_XT_L1_1': stacked_ensemble_XT_L1_1,
                       'stacked_ensemble_XT_L1_2': stacked_ensemble_XT_L1_2,
                       'stacked_ensemble_LGB_L1_1': stacked_ensemble_LGB_L1_1,
                       'stacked_ensemble_LGB_L1_2': stacked_ensemble_LGB_L1_2,
                       'stacked_ensemble_LGB_L1_3': stacked_ensemble_LGB_L1_3,
                       'stacked_ensemble_CAT_L1_1': stacked_ensemble_CAT_L1_1,
                       'stacked_ensemble_CAT_L1_2': stacked_ensemble_CAT_L1_2,
                       }

ensemble_reg_models = {'stacked_ensemble_r__RF_L1': stacked_ensemble_r__RF_L1,
                       'stacked_ensemble_r_XT_L1_1': stacked_ensemble_r_XT_L1_1,
                       'stacked_ensemble_r_XT_L1_2': stacked_ensemble_r_XT_L1_2,
                       'stacked_ensemble_r_LGB_L1_1': stacked_ensemble_r_LGB_L1_1,
                       'stacked_ensemble_r_LGB_L1_2': stacked_ensemble_r_LGB_L1_2,
                       'stacked_ensemble_r_LGB_L1_3': stacked_ensemble_r_LGB_L1_3,
                       'stacked_ensemble_r_CAT_L1_1': stacked_ensemble_r_CAT_L1_1,
                       'stacked_ensemble_r_CAT_L1_2': stacked_ensemble_r_CAT_L1_2,
                       'weighted_ensemble_r__RF_L1': weighted_ensemble_r__RF_L1,
                       'weighted_ensemble_r_XT_L1_1': weighted_ensemble_r_XT_L1_1,
                       'weighted_ensemble_r_XT_L1_2': weighted_ensemble_r_XT_L1_2,
                       'weighted_ensemble_r_LGB_L1_1': weighted_ensemble_r_LGB_L1_1,
                       'weighted_ensemble_r_LGB_L1_2': weighted_ensemble_r_LGB_L1_2,
                       'weighted_ensemble_r_LGB_L1_3': weighted_ensemble_r_LGB_L1_3,
                       'weighted_ensemble_r_CAT_L1_1': weighted_ensemble_r_CAT_L1_1,
                       'weighted_ensemble_r_CAT_L1_2': weighted_ensemble_r_CAT_L1_2
                       }

pretraining_model_choices = ['FlavaFeatureProcessor',
                             'AlbefFeatureProcessor',
                             'FlavaVQA',
                             'AlbefVQA',
                             'FlavaITM',
                             'AlbefITM',
                             'FlavaTextModel',
                             'Data2VecTextModel']

print(f'We will add these {len(downstrem_choices)} downstream models into our config space')

# In[5]:


pretrained_model_path = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/pretrained_model/"
if not os.path.exists(pretrained_model_path):
    os.makedirs(pretrained_model_path)
pretrained_model_dir = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/pretrained_model/"

pretrained_model_path = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/pretrained_model/"

if not os.path.exists(pretrained_model_path):
    os.makedirs(pretrained_model_path)
pretrained_model_dir = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/pretrained_model/"

downstream_model_path = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/downstream_model/"

if not os.path.exists(downstream_model_path):
    os.makedirs(downstream_model_path)
downstream_model_dir = '/mnt/server-home/TUE/20210962/smac3_output/runs' + str(
    seed) + "/downstream_model/"

# loading the csv and constructing tabular + text dataframe
test_df = pd.read_csv('/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/train/train.csv')
test_df['text'] = test_df['Name'] + ", " + test_df['Description']
test_df = test_df.drop(['Name', 'Description'], axis=1)

# Fill na values

dummy_text = test_df['text'][0]
test_df['text'].fillna(dummy_text, inplace=True)

tabular_df = test_df
tabular_df = tabular_df.drop(['text'], axis=1)


class CustomDataset(Dataset):
    def __init__(self, data, captions, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.captions = captions.values.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data[index]
        label = self.captions[index]
        img_path = self.path + img_name + "-" + str(1) + ".jpg"
        try:
            image = Image.open(img_path)
            if self.transform is not None:
                image = self.transform(image)
        except FileNotFoundError as e:
            image = Image.open(
                '/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/test.jpg')
            image = self.transform(image)

        return image, label


img_size = 224
valid_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()])


# defining a custom collate function
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


class DefaultInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates only the default configuration."""

    def _select_configurations(self) -> list[Configuration]:
        config = self._configspace.get_default_configuration()
        config.origin = "Initial Design: Default"
        return [config]


def compute_task_score(predictions, targets):
    # Implement scoring logic here
    # Compute the score based on the predictions and targets of a task
    return accuracy_score(targets, predictions)


# Define the custom late fusion method
def custom_fusion(vision_input, text_input, tabular_input):
    # Custom logic to fuse the multimodal embeddings
    fused_embeddings = torch.cat((vision_input, text_input, tabular_input), dim=1)
    return fused_embeddings


class DefaultInitialDesign(AbstractInitialDesign):
    """Initial design that evaluates only the default configuration."""

    def _select_configurations(self) -> list[Configuration]:
        config = self._configspace.get_default_configuration()
        config.origin = "Initial Design: Default"
        return [config]


def compute_task_score(predictions, targets):
    # Implement scoring logic here
    # Compute the score based on the predictions and targets of a task
    return accuracy_score(targets, predictions)


# Define the custom late fusion method
def custom_fusion(vision_input, text_input, tabular_input):
    # Custom logic to fuse the multimodal embeddings
    fused_embeddings = torch.cat((vision_input, text_input, tabular_input), dim=1)
    return fused_embeddings


def text_data_extractor(dataframe, column_name_list, single_frame_flag):
    idx = 0
    for column in column_name_list:
        globals()['text_str' + str(idx)] = dataframe[[column]]
        idx += 1
    text_df = pd.DataFrame()
    for list_idx in range(len(column_name_list)):
        text_df[column_name_list[list_idx]] = globals()['text_str' + str(list_idx)]
    if single_frame_flag == True:
        text_df = text_df[column_name_list].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    return text_df


def tabular_extractor(dataframe, column_name_list):
    for col in column_name_list:
        dataframe = dataframe.drop([col], axis=1)
    return dataframe


# In[7]:
PATH = '/mnt/server-home/TUE/20210962/petfinder-adoption-prediction/train_images/'
train_dataset = CustomDataset(test_df['PetID'].iloc[:1000], test_df['text'].iloc[:1000], PATH, transform=valid_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
user_defined_task = 'classification'
text_column_list = ['title', 'description']
modality = 'tabular_text_vision'
user_defined_target = 'AdoptionSpeed'
classification_objective = 'multiclass'
num_classes = 5

global device, fine_tuning_model, fine_tuning_model_stacked, max_depth, tabular_embeds
import signal
# Define a handler function for the timeout signal
def handler(signum, frame):
    raise TimeoutError("Model search process reached the time limit")


access_token = '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'
login(token=access_token)
flava_pretraining_model = FlavaForPreTraining.from_pretrained("facebook/flava-full", use_auth_token=access_token)
flava_processor = FlavaProcessor.from_pretrained("facebook/flava-full", use_auth_token=access_token)  #

# Now half NAS technique to generate Tabular fit
#############**********************########################
if tabular_df[user_defined_target].min() < 0:
    tabular_df[user_defined_target] += abs(tabular_df[user_defined_target].min())


tabular_embeds = None
if modality == 'tabular_text_vision' or 'tabular_text':
    print(f'The selected modality is {modality} \n')
    sys.stdout.flush()
    print(f'Starting NAS for the Tabular Modality \n')
    sys.stdout.flush()
    tabular_predictor = MultiModalPredictor(label='AdoptionSpeed')
    tabular_predictor.fit(tabular_df.iloc[:1000])
    tabular_embeds = tabular_predictor.extract_embedding(tabular_df.iloc[:1000])

def objective_function(config, seed):
    import torch
    device = 'cuda'
    coffee = '\u2615'
    global objective_value, fine_tuning_model_stacked, fine_tuning_model
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Extract the hyperparameters from the configuration
    # Extract the hyperparameters from the configuration
    pretraining_model = config['pretraining_model']
    try:
        pretraining_task = config['pretraining_task']  # ---------User Argument
    except KeyError:
        pretraining_task = 'regression'
    downstream_model = config['downstream_model']
    downstream_task = config['downstream_task']

    downstream_feature_processor = AutoMLPipelineFeatureGenerator()
    # We load the pretrained model str instance from the config space, check what it is and then load respective models

    ## TABULAR + TEXT + VISION Modality
    if modality == 'tabular_text_vision':
        ###KEEP AN EYE FOR THE Tabular_df variable in line 165

        if user_defined_task == 'classification' and pretraining_task == 'classification':

            ##General statement that will appear when the search for the classification task starts###
            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the classification task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()

            if pretraining_model == 'FlavaFeatureProcessor':  ##extension for Albef is to be added
                pretraining_model = flava_pretraining_model
                pretraining_feature_processor = flava_processor

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config).cuda()
                pretraining_feature_processor = pretraining_feature_processor

                target = user_defined_target  ## --- This will be a user argument

                text_img_embeddings = []
                for idx, batch in tqdm(enumerate(train_loader)):
                    img, text = batch
                    converted_tensors = [
                        torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                        for
                        tensor in img]

                    for itr in tqdm(range(batch_size)):
                        inputs = pretraining_feature_processor(text=str(text[itr][:512]),
                                                               images=[converted_tensors[itr].convert("RGB")],
                                                               return_tensors="pt", max_length=77, padding=True,
                                                               return_codebook_pixels=True,
                                                               return_image_mask=True).to(device) # cuda
                        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
                        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
                        outputs = pretraining_model(**inputs)

                        mm_embedding = outputs.multimodal_masked_output.last_hidden_state

                        ##Linear Layer :
                        Linear_Layer = nn.Linear(mm_embedding.size()[2], pretraining_linear_hidden_size, device = device)
                        mm_embeddings = Linear_Layer(mm_embedding)
                        # here we get the text and image embeddings
                        text_img_embeddings.append(mm_embeddings.cpu().detach().numpy().flatten())  # Flatten the multimodal embedding

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "flava_model" + str(seed))

                ##**NAS##
                # tab_embeds = tab_embeds * np.random.randint(3, 365)  # tabular_predictor.extract_embedding(tabular_df
                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(tab_embeds[0].size()[0], pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, text_img_embeddings)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_df[user_defined_target].iloc[:1000]], axis=1)

                final_stacker_target = final_stacker[user_defined_target]
                final_stacker_data = final_stacker.drop([user_defined_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        if classification_objective == 'multiclass':
                            fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth,
                                                                   loss_function='MultiClass',
                                                                   classes_count=num_classes)
                        else:
                            fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth,
                                                                   loss_function='CrossEntropy')

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                    if downstream_model == 'XGBoostClassifier':
                        try:
                            print('XGBoostClassifier as DTM')

                            xgboost_max_depth = config['xgboost_max_depth']

                            xgboost_num_boost_round = config['xgboost_num_boost_round']
                            print(xgboost_num_boost_round)

                            grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}

                            fine_tuning_model = XGBClassifier()
                            fine_tuning_model.set_params(**grid)

                            seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                              "feature_processor": 'AutoMlFeatureGenerator',
                                                              "xgboost_max_depth": xgboost_max_depth,
                                                              "xgboost_num_boost_round": xgboost_num_boost_round,
                                                              "downstream_task": config['downstream_task']
                                                              }}

                            # Train the downstream model
                            fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                            valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                            # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                            downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                            if not os.path.exists(downstream_model_path_seed):
                                os.makedirs(downstream_model_path_seed)
                            downstream_model_path_seed_dir = downstream_model_path_seed

                            # Write the data to the JSON file
                            with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                      "w") as json_file:
                                json.dump(seed_config_dict_ds, json_file, indent=4)

                            ####Save the models here############

                            fine_tuning_model.save_model(
                                downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                            score = accuracy_score(valid_predictions, y_test)
                            # dummy_score = random.uniform(0,1)

                            objective_value = score
                            print(f'The computed objective value is {objective_value}')
                        except KeyError:
                            pass

                    if downstream_model in list(ensemble_clf_models.keys()):
                        print(f'Selected Ensemble Model is {downstream_model}')
                        # fine_tuning_model_stacked = None
                        tuple_list = []
                        for key, value in ensemble_clf_models.items():
                            if downstream_model == key:
                                for model_tuple in value.estimators:
                                    if model_tuple[0] == 'LGBClassifier':
                                        num_leaves = config['lgbm_num_leaves']
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth)
                                        print('Hyperparam Set')
                                    elif model_tuple[0] == 'XGBoostClassifier':
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                    elif model_tuple[0] == 'CatBoostClassifier':
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        ##cat boost has slightly wierd objectives. we can correct those here
                                        ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'KNNClassifier':
                                        n_neighbors = config['knn_n_neighbors']
                                        model_tuple[1].set_params(n_neighbors=n_neighbors)
                                        print('Hyperparameter Set')
                                fine_tuning_model_stacked = value

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "max_depth": max_depth,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        # fine_tuning_model_stacked.save_model(
                        #   downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        sys.stdout.flush()
                        print(f'The computed objective value is {objective_value}')

            if pretraining_model == 'AlbefFeatureProcessor':
                config_file_for_albef = yaml.load(
                    open("/mnt/server-home/TUE/20210962/optimisation/multimodal/examples/albef/configs/retrieval.yaml",
                         "r"), Loader=yaml.Loader)
                albef_pretraining_model = albef_model_for_retrieval(config_file_for_albef)
                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']


                # vision encoder parameters
                config_file_for_albef['hidden_size'] = pretraining_hidden_size
                config_file_for_albef['embed_size'] = pretraining_linear_hidden_size
                config_file_for_albef['vision_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['vision_encoder_args']['hidden_size'] = pretraining_hidden_size
                # text encoder parameters
                config_file_for_albef['text_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['text_encoder_args']['hidden_size'] = pretraining_hidden_size

                # multimodal encoder parameters
                config_file_for_albef['multimodal_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['multimodal_encoder_args']['hidden_size'] = pretraining_hidden_size

                config_file_for_albef['projection_args']['in_features'] = pretraining_hidden_size

                ##Linear Layer Size
                config_file_for_albef['projection_args']['out_features'] = pretraining_linear_hidden_size
                config_file_for_albef['similarity_args']['embed_size'] = pretraining_linear_hidden_size

                albef_pretraining_model = albef_model_for_retrieval(config)

                target = user_defined_target  ## --- This will be a user argument
                ## This is processing the entire batch
                image_text_embedding_list, text_embedding_list = [], []
                for idx, batch in enumerate(train_loader):
                    image_batch, text_batch = batch
                    images = [torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                              for
                              tensor in image_batch]
                    image_input = [image_transform(image) for image in images]
                    image_input = torch.stack(image_input, dim=0)
                    text_batch = list(text_batch)
                    text_input = [text_transform(text) for text in text_batch]
                    text_attention_mask = [(text_input_itr != 0).type(torch.long) for text_input_itr in text_input]
                    text_input = torch.stack(text_input, dim=0)
                    text_attention_mask = torch.stack(text_attention_mask, dim=0)

                    image_embed, image_feat = albef_pretraining_model(image=image_input, input_type="image", is_train=False)
                    text_embed, text_feat = albef_pretraining_model(text=text_input, text_atts=text_attention_mask, input_type="text",
                                                  is_train=False)

                    image_text_embeds = torch.cat((image_embed, text_embed), dim=1)
                    image_text_embeds = torch.flatten(image_text_embeds, start_dim=1, end_dim=2)
                    Linear_Layer = nn.Linear(image_text_embeds.size()[1], linear_dim)  ##Linear Layer
                    image_text_embeds = Linear_Layer(image_text_embeds)
                    image_text_embedding_list.append(image_text_embeds)

                    image_text_similarity_score = albef_pretraining_model._image_text_matching_score(image=image_embed,
                                                                                     text=text_embed,
                                                                          text_atts=text_attention_mask).max().item()

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "albef_model" + str(seed))

                ##**NAS##
                # tab_embeds = tab_embeds * np.random.randint(3, 365)  # tabular_predictor.extract_embedding(tabular_df
                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(tab_embeds[0].size()[0], pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, image_text_embedding_list)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_df[user_defined_target].iloc[:1000]], axis=1)

                final_stacker_target = final_stacker[user_defined_target]
                final_stacker_data = final_stacker.drop([user_defined_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        if classification_objective == 'multiclass':
                            fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth,
                                                                   loss_function='MultiClass',
                                                                   classes_count=num_classes)
                        else:
                            fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth,
                                                                   loss_function='CrossEntropy')

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                    if downstream_model == 'XGBoostClassifier':
                        try:
                            print('XGBoostClassifier as DTM')

                            xgboost_max_depth = config['xgboost_max_depth']

                            xgboost_num_boost_round = config['xgboost_num_boost_round']
                            print(xgboost_num_boost_round)

                            grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}

                            fine_tuning_model = XGBClassifier()
                            fine_tuning_model.set_params(**grid)

                            seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                              "feature_processor": 'AutoMlFeatureGenerator',
                                                              "xgboost_max_depth": xgboost_max_depth,
                                                              "xgboost_num_boost_round": xgboost_num_boost_round,
                                                              "downstream_task": config['downstream_task']
                                                              }}

                            # Train the downstream model
                            fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                            valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                            # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                            downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                            if not os.path.exists(downstream_model_path_seed):
                                os.makedirs(downstream_model_path_seed)
                            downstream_model_path_seed_dir = downstream_model_path_seed

                            # Write the data to the JSON file
                            with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                      "w") as json_file:
                                json.dump(seed_config_dict_ds, json_file, indent=4)

                            ####Save the models here############

                            fine_tuning_model.save_model(
                                downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                            score = accuracy_score(valid_predictions, y_test)
                            # dummy_score = random.uniform(0,1)

                            objective_value = score
                            print(f'The computed objective value is {objective_value}')
                        except KeyError:
                            pass

                    if downstream_model in list(ensemble_clf_models.keys()):
                        print(f'Selected Ensemble Model is {downstream_model}')
                        # fine_tuning_model_stacked = None
                        tuple_list = []
                        for key, value in ensemble_clf_models.items():
                            if downstream_model == key:
                                for model_tuple in value.estimators:
                                    if model_tuple[0] == 'LGBClassifier':
                                        num_leaves = config['lgbm_num_leaves']
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth)
                                        print('Hyperparam Set')
                                    elif model_tuple[0] == 'XGBoostClassifier':
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                    elif model_tuple[0] == 'CatBoostClassifier':
                                        try:
                                            max_depth = 5
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        ##cat boost has slightly wierd objectives. we can correct those here
                                        ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'KNNClassifier':
                                        n_neighbors = config['knn_n_neighbors']
                                        model_tuple[1].set_params(n_neighbors=n_neighbors)
                                        print('Hyperparameter Set')
                                fine_tuning_model_stacked = value

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "max_depth": max_depth,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        # fine_tuning_model_stacked.save_model(
                        #   downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        sys.stdout.flush()
                        print(f'The computed objective value is {objective_value}')


        if user_defined_task == 'regression' and pretraining_task == 'regression':
            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the regression task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()

            if pretraining_model == 'FlavaFeatureProcessor':  ##extension for Albef is to be added

                pretraining_model = FlavaForPreTraining.from_pretrained("facebook/flava-full",
                                                                        use_auth_token=access_token)
                pretraining_feature_processor = FlavaProcessor.from_pretrained("facebook/flava-full",
                                                                               use_auth_token=access_token)  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                target = user_defined_target  ## --- This will be a user argument

                text_img_embeddings = []
                for idx, batch in tqdm(enumerate(train_loader)):
                    img, text = batch
                    converted_tensors = [
                        torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                        for
                        tensor in img]

                    for itr in tqdm(range(batch_size)):
                        inputs = pretraining_feature_processor(text=str(text[itr][:512]),
                                                               images=[converted_tensors[itr].convert("RGB")],
                                                               return_tensors="pt", max_length=77, padding=True,
                                                               return_codebook_pixels=True,
                                                               return_image_mask=True)  # cuda
                        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
                        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
                        outputs = pretraining_model(**inputs)

                        mm_embedding = outputs.multimodal_masked_output.last_hidden_state

                        ##Linear Layer :
                        Linear_Layer = nn.Linear(mm_embedding.size()[2], pretraining_linear_hidden_size)

                        mm_embeddings = Linear_Layer(mm_embedding)

                        # here we get the text and image embeddings
                        text_img_embeddings.append(
                            mm_embedding.detach().numpy().flatten())  # Flatten the multimodal embedding

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "flava_model" + str(seed))

                # Now half NAS technique to generate Tabular fit
                #############**********************########################
                #         if tabular_df['AdoptionSpeed'].min() < 0:
                #             tabular_df['AdoptionSpeed'] += abs(tabular_df['AdoptionSpeed'].min())

                #         tabular_predictor = MultiModalPredictor(label='AdoptionSpeed',problem_type= 'multiclass')
                #         tabular_predictor.fit(tabular_df)
                #         tab_embeds = tabular_predictor.extract_embeddings(tabular_df)

                #############**********************########################

                # tab_embeds = np.random.randn(50, 144)  # tabular_predictor.extract_embedding(tabular_df
                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(tab_embeds[0].size()[0], pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, text_img_embeddings)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_df[user_defined_target]], axis=1)

                final_stacker_target = final_stacker[user_defined_target]
                final_stacker_data = final_stacker.drop([user_defined_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Set')
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    problem = 'regression'
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                            fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model_stacked.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')
            if pretraining_model == 'AlbefFeatureProcessor':
                config_file_for_albef = yaml.load(
                    open("/mnt/server-home/TUE/20210962/optimisation/multimodal/examples/albef/configs/retrieval.yaml",
                         "r"), Loader=yaml.Loader)
                albef_pretraining_model = albef_model_for_retrieval(config_file_for_albef)
                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # vision encoder parameters
                config_file_for_albef['hidden_size'] = pretraining_hidden_size
                config_file_for_albef['embed_size'] = pretraining_linear_hidden_size
                config_file_for_albef['vision_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['vision_encoder_args']['hidden_size'] = pretraining_hidden_size
                # text encoder parameters
                config_file_for_albef['text_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['text_encoder_args']['hidden_size'] = pretraining_hidden_size

                # multimodal encoder parameters
                config_file_for_albef['multimodal_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['multimodal_encoder_args']['hidden_size'] = pretraining_hidden_size

                config_file_for_albef['projection_args']['in_features'] = pretraining_hidden_size

                ##Linear Layer Size
                config_file_for_albef['projection_args']['out_features'] = pretraining_linear_hidden_size
                config_file_for_albef['similarity_args']['embed_size'] = pretraining_linear_hidden_size

                albef_pretraining_model = albef_model_for_retrieval(config)

                target = user_defined_target  ## --- This will be a user argument
                ## This is processing the entire batch
                image_text_embedding_list, text_embedding_list = [], []
                for idx, batch in enumerate(train_loader):
                    image_batch, text_batch = batch
                    images = [torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                              for
                              tensor in image_batch]
                    image_input = [image_transform(image) for image in images]
                    image_input = torch.stack(image_input, dim=0)
                    text_batch = list(text_batch)
                    text_input = [text_transform(text) for text in text_batch]
                    text_attention_mask = [(text_input_itr != 0).type(torch.long) for text_input_itr in text_input]
                    text_input = torch.stack(text_input, dim=0)
                    text_attention_mask = torch.stack(text_attention_mask, dim=0)

                    image_embed, image_feat = albef_pretraining_model(image=image_input, input_type="image",
                                                                      is_train=False)
                    text_embed, text_feat = albef_pretraining_model(text=text_input, text_atts=text_attention_mask,
                                                                    input_type="text",
                                                                    is_train=False)

                    image_text_embeds = torch.cat((image_embed, text_embed), dim=1)
                    image_text_embeds = torch.flatten(image_text_embeds, start_dim=1, end_dim=2)
                    Linear_Layer = nn.Linear(image_text_embeds.size()[1], linear_dim)  ##Linear Layer
                    image_text_embeds = Linear_Layer(image_text_embeds)
                    image_text_embedding_list.append(image_text_embeds)

                    image_text_similarity_score = albef_pretraining_model._image_text_matching_score(image=image_embed,
                                                                                                     text=text_embed,
                                                                                                     text_atts=text_attention_mask).max().item()

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "albef_model" + str(seed))

                ##**NAS##
                # tab_embeds = tab_embeds * np.random.randint(3, 365)  # tabular_predictor.extract_embedding(tabular_df
                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(tab_embeds[0].size()[0], pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, image_text_embedding_list)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_df[user_defined_target].iloc[:1000]], axis=1)

                final_stacker_target = final_stacker[user_defined_target]
                final_stacker_data = final_stacker.drop([user_defined_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Set')
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    problem = 'regression'
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    try:
                                        max_depth = config['lgbm_num_leaves']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                            fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model_stacked.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')

    ####################*****************************************####################
    # TEXT + VISION Modality
    if modality == 'text_vision':
        ## VQA TASK STARTS####
        if user_defined_task == 'VQA' and pretraining_task == 'VQA':

            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the VQA task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()
            if pretraining_model == 'FlavaVQA':
                max_length = 512
                batch_loss = []
                predictions = []
                batch_embeddings = []
                iter_embeddings = []
                mm_embeddings = []
                prediction_list = []

                pretraining_model = FlavaForPreTraining.from_pretrained("facebook/flava-full",
                                                                        use_auth_token=access_token)
                pretraining_feature_processor = FlavaProcessor.from_pretrained("facebook/flava-full",
                                                                               use_auth_token=access_token)  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                answer_candidates = [1, 0]
                flava_class_head = flava_model_for_classification(num_classes=len(answer_candidates))
                argmax_func = lambda x: np.argmax(x)

                for idx, batch in enumerate(train_loader):
                    img, question, answer = batch
                    question = list(question)
                    converted_batch_tensors = []
                    convert_to_PIL = transforms.ToPILImage()

                    img_tensor_list = []
                    for idx in tqdm(range(batch_size)):
                        img_tensor = img[idx][:3].to('cuda')
                        img_tensor_list.append(img_tensor)
                    img_tensor_final = torch.stack((img_tensor_list))
                    img_tensor_final = F.interpolate(img_tensor_final,
                                                     size=224)  # model requires dimension to be 224 x 224
                    img_tensor_final = img_tensor_final
                    print(img_tensor_final.size())
                    print('Starting VQA..\n')

                    tokenized_question = bert_tokenizer(question)
                    # Calculating the max sequence length after max pooling
                    len_ = [len(arr) for arr in tokenized_question['input_ids']]
                    max_len = max(len_)
                    padded_token_list = [list_element + [0] * max(0, max_len - len(list_element)) for list_element
                                         in
                                         tokenized_question['input_ids']]
                    question_tensor = torch.tensor(padded_token_list)

                    vqa_outputs = flava_class_head(text=question_tensor, image=img_tensor_final.cpu(),
                                                   labels=torch.as_tensor(answer))

                    # collecting the batch loss
                    batch_loss.append(vqa_outputs.loss)
                    # collecting logits and the predictions ('yes' or 'no' : 'yes = 1', 'no = 0')
                    predicted_logits = vqa_outputs.logits.detach().cpu().numpy()
                    predicted_labels = [argmax_func(itm) for itm in predicted_logits]
                    prediction_list.append(predicted_labels)
                    # Collecting the embeddings from the feature processor
                    print('Extracting Embeddings..\n')
                    # converting tensor to images
                    tens_to_img = [convert_to_PIL(item) for item in img_tensor_final]
                    inputs = pretraining_feature_processor(text=question, images=tens_to_img, return_tensors="pt",
                                                           max_length=77,
                                                           padding=True, return_codebook_pixels=True,
                                                           return_image_mask=True)
                    # cuda
                    inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
                    inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
                    inputs = inputs.to('cuda')
                    outputs = pretraining_model(**inputs)
                    mm_embedding = outputs.multimodal_masked_output.last_hidden_state.detach().cpu()
                    mm_embeddings.append(mm_embedding)
                    print('Done Extraction..\n')
                    # collect all the predictions

                final_predictions = []
                for arr in prediction_list:
                    for pred_label in arr:
                        final_predictions.append(pred_label)

                # print(len(mm_embeddings))
                flat_tensors = []
                for batch_embeds in mm_embeddings:
                    for inst_embeds in batch_embeds:
                        flat_tensors.append(inst_embeds.flatten())

                # Pooling the embeddings

                max_pool = nn.MaxPool1d(4, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in flat_tensors]

                # flat_embed_list = [item.numpy().tolist() for item in pool_embedding]

                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')

                pooled_embeddings_final = pooled_embeddings_final.tolist()

                dict_1 = {"embeddings": pooled_embeddings_final}
                dict_2 = {"embeddings": pooled_embeddings_final, "targets": final_predictions}

                gluon_df_1 = pd.DataFrame(dict_1)
                gluon_df_1 = pd.concat([gluon_df_1[col].apply(pd.Series) for col in gluon_df_1.columns], axis=1,
                                       ignore_index=True)

                gluon_df_2 = pd.DataFrame(dict_2)

                gluon_df_1['targets'] = gluon_df_2['targets']

                # gluon_df_1 is the final dataframe that needs to be paased
                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(gluon_df_1.iloc[:, :-1], gluon_df_1['targets'],
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth)

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                if downstream_model == 'XGBoostClassifier':
                    try:
                        print('XGBoostClassifier as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBClassifier()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass

                if downstream_model in list(ensemble_clf_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_clf_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBClassifier':
                                    objective_loss = 'binary'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=objective_loss)
                                    print('Hyperparam Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostClassifier':
                                    objective_loss = 'binary'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=objective_loss)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    objective_loss = 'CrossEntropy'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, objective=objective_loss)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'KNNClassifier':
                                    n_neighbors = config['knn_n_neighbors']
                                    model_tuple[1].set_params(n_neighbors=n_neighbors)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model_stacked.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = accuracy_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')

            if pretraining_model == 'AlbefVQA':
                config_file_for_albef = yaml.load(
                    open("/mnt/server-home/TUE/20210962/optimisation/multimodal/examples/albef/configs/retrieval.yaml",
                         "r"), Loader=yaml.Loader)
                albef_pretraining_model = albef_model_for_retrieval(config_file_for_albef)
                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # vision encoder parameters
                config_file_for_albef['hidden_size'] = pretraining_hidden_size
                config_file_for_albef['embed_size'] = pretraining_linear_hidden_size
                config_file_for_albef['vision_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['vision_encoder_args']['hidden_size'] = pretraining_hidden_size
                # text encoder parameters
                config_file_for_albef['text_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['text_encoder_args']['hidden_size'] = pretraining_hidden_size

                # multimodal encoder parameters
                config_file_for_albef['multimodal_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['multimodal_encoder_args']['hidden_size'] = pretraining_hidden_size

                config_file_for_albef['projection_args']['in_features'] = pretraining_hidden_size

                ##Linear Layer Size
                config_file_for_albef['projection_args']['out_features'] = pretraining_linear_hidden_size
                config_file_for_albef['similarity_args']['embed_size'] = pretraining_linear_hidden_size

                albef_pretraining_model = albef_model_for_retrieval(config)

                target = user_defined_target  ## --- This will be a user argument
                ## This is processing the entire batch
                image_text_embedding_list, text_embedding_list = [], []
                prediction_list = []
                for idx, batch in enumerate(train_loader):
                    image_batch, text_batch = batch
                    images = [torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                              for
                              tensor in image_batch]
                    image_input = [image_transform(image) for image in images]
                    image_input = torch.stack(image_input, dim=0)
                    text_batch = list(text_batch)
                    text_input = [text_transform(text) for text in text_batch]
                    text_attention_mask = [(text_input_itr != 0).type(torch.long) for text_input_itr in text_input]
                    text_input = torch.stack(text_input, dim=0)
                    text_attention_mask = torch.stack(text_attention_mask, dim=0)

                    image_embed, image_feat = albef_pretraining_model(image=image_input, input_type="image",
                                                                      is_train=False)
                    text_embed, text_feat = albef_pretraining_model(text=text_input, text_atts=text_attention_mask,
                                                                    input_type="text",
                                                                    is_train=False)

                    image_text_embeds = torch.cat((image_embed, text_embed), dim=1)
                    image_text_embeds = torch.flatten(image_text_embeds, start_dim=1, end_dim=2)
                    Linear_Layer = nn.Linear(image_text_embeds.size()[1], linear_dim)  ##Linear Layer
                    image_text_embeds = Linear_Layer(image_text_embeds)
                    image_text_embedding_list.append(image_text_embeds)

                    image_text_similarity_score = albef_pretraining_model._image_text_matching_score(image=image_embed,
                                                                                                     text=text_embed,
                                                                 text_atts=text_attention_mask).argmax().item()
                    prediction_list.append(image_text_similarity_score)

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "albef_model" + str(seed))
                pooled_embeddings_final = np.array(image_text_embedding_list, dtype=object)
                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                pooled_embeddings_final = pooled_embeddings_final.tolist()

                dict_1 = {"embeddings": pooled_embeddings_final}
                dict_2 = {"embeddings": pooled_embeddings_final, "targets": test_df[user_defined_target].iloc[:1000]}

                gluon_df_1 = pd.DataFrame(dict_1)
                gluon_df_1 = pd.concat([gluon_df_1[col].apply(pd.Series) for col in gluon_df_1.columns], axis=1,
                                       ignore_index=True)

                gluon_df_2 = pd.DataFrame(dict_2)

                gluon_df_1['targets'] = gluon_df_2['targets']

                # gluon_df_1 is the final dataframe that needs to be paased
                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(gluon_df_1.iloc[:, :-1], gluon_df_1['targets'],
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth)

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                if downstream_model == 'XGBoostClassifier':
                    try:
                        print('XGBoostClassifier as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBClassifier()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass

                if downstream_model in list(ensemble_clf_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_clf_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBClassifier':
                                    objective_loss = 'binary'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=objective_loss)
                                    print('Hyperparam Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostClassifier':
                                    objective_loss = 'binary'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=objective_loss)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    objective_loss = 'CrossEntropy'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, objective=objective_loss)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'KNNClassifier':
                                    n_neighbors = config['knn_n_neighbors']
                                    model_tuple[1].set_params(n_neighbors=n_neighbors)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = accuracy_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')

        ###ITM TASK STARTS################
        if user_defined_task == 'ITM':
            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the ITM task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()
            if pretraining_model == 'FlavaITM':
                embeddings = []
                item_scores = []

                pretraining_model = FlavaForPreTraining.from_pretrained("facebook/flava-full",
                                                                        use_auth_token=access_token)
                pretraining_feature_processor = FlavaProcessor.from_pretrained("facebook/flava-full",
                                                                               use_auth_token=access_token)  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor
                for idx, batch in tqdm(enumerate(train_loader)):
                    img, text = batch
                    converted_tensors = [
                        torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                        for
                        tensor in img]

                    for itr in tqdm(range(batch_size)):
                        inputs = pretraining_feature_processor(text=str(text[itr][:512]),
                                                               images=[converted_tensors[itr].convert("RGB")],
                                                               return_tensors="pt", max_length=77, padding=True,
                                                               return_codebook_pixels=True,
                                                               return_image_mask=True)  # cuda
                        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
                        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
                        outputs = pretraining_model(**inputs)
                        mm_embedding = outputs.multimodal_masked_output.last_hidden_state

                        flava_contrastive_scores = outputs.contrastive_logits_per_image.item()
                        flava_itm_scores = torch.nn.functional.softmax(outputs.itm_logits)[0][1].item()
                        item_scores.append(flava_contrastive_scores)

                        ##Linear Layer :
                        Linear_Layer = nn.Linear(mm_embedding.size()[2], pretraining_linear_hidden_size)
                        mm_embeddings = Linear_Layer(mm_embedding)
                        # here we get the text and image embeddings
                        embeddings.append(
                            mm_embedding.detach().numpy().flatten())  # Flatten the multimodal embedding

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "flava_model" + str(seed))

                dict_ = {'embeddings': embeddings, 'targets': item_scores}

                embeddings_df = pd.DataFrame(dict_)
                embeddings_col = pd.DataFrame(embeddings_df['embeddings'], columns=['embeddings'])

                tensor_embedding = [torch.Tensor(emx) for emx in
                                    embeddings_col['embeddings']]  # Converting to tensors to apply pooling

                # Max Pooling the embeddings
                print('Starting Pooling..\n')
                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                print('Done Pooling..\n')
                sys.stdout.flush()
                train_stacker_df = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                embedding_dataframe = pd.concat(
                    [train_stacker_df[col].apply(pd.Series) for col in train_stacker_df.columns], axis=1,
                    ignore_index=True)
                embedding_dataframe['targets'] = embeddings_df['targets']

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(embedding_dataframe.iloc[:, :-1],
                                                                    embedding_dataframe['targets'],
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))

                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    problem = 'regression'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')

            if pretraining_model == 'AlbefITM':
                config_file_for_albef = yaml.load(
                    open("/mnt/server-home/TUE/20210962/optimisation/multimodal/examples/albef/configs/retrieval.yaml",
                         "r"), Loader=yaml.Loader)
                albef_pretraining_model = albef_model_for_retrieval(config_file_for_albef)
                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # vision encoder parameters
                config_file_for_albef['hidden_size'] = pretraining_hidden_size
                config_file_for_albef['embed_size'] = pretraining_linear_hidden_size
                config_file_for_albef['vision_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['vision_encoder_args']['hidden_size'] = pretraining_hidden_size
                # text encoder parameters
                config_file_for_albef['text_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['text_encoder_args']['hidden_size'] = pretraining_hidden_size

                # multimodal encoder parameters
                config_file_for_albef['multimodal_encoder_args']['num_hidden_layers'] = pretraining_num_layers
                config_file_for_albef['multimodal_encoder_args']['hidden_size'] = pretraining_hidden_size

                config_file_for_albef['projection_args']['in_features'] = pretraining_hidden_size

                ##Linear Layer Size
                config_file_for_albef['projection_args']['out_features'] = pretraining_linear_hidden_size
                config_file_for_albef['similarity_args']['embed_size'] = pretraining_linear_hidden_size

                albef_pretraining_model = albef_model_for_retrieval(config)

                target = user_defined_target  ## --- This will be a user argument
                ## This is processing the entire batch
                image_text_embedding_list, text_embedding_list = [], []
                prediction_list = []
                for idx, batch in enumerate(train_loader):
                    image_batch, text_batch = batch
                    images = [torchvision.transforms.functional.to_pil_image(tensor, mode=None).convert("RGB")
                              for
                              tensor in image_batch]
                    image_input = [image_transform(image) for image in images]
                    image_input = torch.stack(image_input, dim=0)
                    text_batch = list(text_batch)
                    text_input = [text_transform(text) for text in text_batch]
                    text_attention_mask = [(text_input_itr != 0).type(torch.long) for text_input_itr in text_input]
                    text_input = torch.stack(text_input, dim=0)
                    text_attention_mask = torch.stack(text_attention_mask, dim=0)

                    image_embed, image_feat = albef_pretraining_model(image=image_input, input_type="image",
                                                                      is_train=False)
                    text_embed, text_feat = albef_pretraining_model(text=text_input, text_atts=text_attention_mask,
                                                                    input_type="text",
                                                                    is_train=False)

                    image_text_embeds = torch.cat((image_embed, text_embed), dim=1)
                    image_text_embeds = torch.flatten(image_text_embeds, start_dim=1, end_dim=2)
                    Linear_Layer = nn.Linear(image_text_embeds.size()[1], linear_dim)  ##Linear Layer
                    image_text_embeds = Linear_Layer(image_text_embeds)
                    image_text_embedding_list.append(image_text_embeds)

                    image_text_similarity_score = albef_pretraining_model._image_text_matching_score(image=image_embed,
                                                                                                     text=text_embed,
                                                                 text_atts=text_attention_mask).max().item()
                    prediction_list.append(image_text_similarity_score)

                pretrained_model_path_seed = pretrained_model_path + str(seed) + "/"

                if not os.path.exists(pretrained_model_path_seed):
                    os.makedirs(pretrained_model_path_seed)
                pretrained_model_seed_dir = pretrained_model_path_seed

                # writing to a config_file
                seed_config_dict = {"config": {"pretraining_model": config['pretraining_model'],
                                               "pretraining_feature_processor": str(pretraining_feature_processor),
                                               "pretraining_num_layers": config['pretraining_num_layers'],
                                               "pretraining_hidden_size": config['pretraining_hidden_size'],
                                               "pretraining_linear_hidden_size": config[
                                                   'pretraining_linear_hidden_size'],
                                               "pretraining_pooling_kernel": config['pretraining_pooling_kernel'],
                                               "pretraining_task": user_defined_task
                                               }}

                # Write the data to the JSON file
                with open(pretrained_model_seed_dir + 'pretrained_model_config.json', "w") as json_file:
                    json.dump(seed_config_dict, json_file, indent=4)

                torch.save(pretraining_model.state_dict(), pretrained_model_seed_dir + "albef_model" + str(seed))
                pooled_embeddings_final = np.array(image_text_embedding_list, dtype=object)
                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                pooled_embeddings_final = pooled_embeddings_final.tolist()

                dict_1 = {"embeddings": pooled_embeddings_final}
                dict_2 = {"embeddings": pooled_embeddings_final, "targets": test_df[user_defined_target].iloc[:1000]}

                gluon_df_1 = pd.DataFrame(dict_1)
                gluon_df_1 = pd.concat([gluon_df_1[col].apply(pd.Series) for col in gluon_df_1.columns], axis=1,
                                       ignore_index=True)

                gluon_df_2 = pd.DataFrame(dict_2)

                gluon_df_1['targets'] = gluon_df_2['targets']

                # gluon_df_1 is the final dataframe that needs to be paased
                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(gluon_df_1.iloc[:, :-1], gluon_df_1['targets'],
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    problem = 'regression'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')
    ## TABULAR + TEXT Modality######
    if modality == 'tabular_text':
        if user_defined_task and pretraining_task == 'classification':
            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the classification task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()
            if pretraining_model == 'FlavaTextModel':  ##extension for Albef is to be added
                pretraining_model = FlavaTextModel.from_pretrained("facebook/flava-full")
                pretraining_feature_processor = AutoTokenizer.from_pretrained("facebook/flava-full")  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                tabular_target = user_defined_target  ## --- This will be a user argument

                # """ Text Processor"""
                ##Let us build the text dataset extractor -- IP DATA = test_df

                raw_text_data = text_data_extractor(test_df.data, text_column_list, single_frame_flag=True)
                tabular_data = tabular_extractor(test_df.data, text_column_list)
                print(f'The shape of the text data is {raw_text_data.shape}')
                print(f'The shape of the tabular data is {tabular_data.shape}')

                # text_data = raw_text_data.tolist()

                # Truncating our text sequences
                text_data = []
                truncate = lambda x: x[:512]
                for idx in range(len(raw_text_data)):
                    text_data.append(truncate(raw_text_data[idx]))

                print('Tokenizing Text..\n')
                sys.stdout.flush()

                inputs = []
                device = 'cuda'
                for text in text_data:
                    inputs.append(pretraining_feature_processor(text, return_tensors="pt").to(device))

                ## Token Processing (for token sequences greater than 512) : We need to process these sequences somehow
                new_inputs = []
                for i in range(len(inputs)):
                    if inputs[i]['input_ids'].size()[1] > 512:
                        print(f'The idx {i} has > 512 len')
                        tensor_list, token_type_list, attention_mask_list = [], [], []
                        for tidx in range(511):
                            tensor_list.append(inputs[i]['input_ids'][0][tidx])
                            token_type_list.append(inputs[i]['token_type_ids'][0][tidx])
                            attention_mask_list.append(inputs[i]['attention_mask'][0][tidx])

                            tensor = torch.cuda.IntTensor(tensor_list)
                            token_type_ids = torch.IntTensor(token_type_list)
                            attention_mask = torch.IntTensor(attention_mask_list)

                        inputs[i]['input_ids'] = torch.unsqueeze(tensor, 0)
                        inputs[i]['token_type_ids'] = torch.unsqueeze(token_type_ids, 0)
                        inputs[i]['attention_mask'] = torch.unsqueeze(attention_mask, 0)
                        new_inputs.append(inputs[i])

                outputs = []
                for i in range(len(inputs)):
                    print(f'ITR {i}')
                    sys.stdout.flush()
                    model_output_embedding = pretraining_model(**inputs[i].to(device)).last_hidden_state
                    ##Linear Layer :
                    Linear_Layer = nn.Linear(model_output_embedding.size()[2], pretraining_linear_hidden_size)
                    mm_embeddings = Linear_Layer(model_output_embedding)
                    # here we get the text and image embeddings
                    outputs.append(mm_embeddings.detach().numpy().flatten())  # Flatten the multimodal

                print('Done Tokenizing... \n')
                sys.stdout.flush()
                # import torch
                # torch.cuda.get_arch_list()
                print('Getting Tabular Embeddings \n')
                sys.stdout.flush()

                ######*******HALF NAS*******######
                # from autogluon.multimodal import MultiModalPredictor
                #
                # predictor = MultiModalPredictor(label='fraudulent')
                # predictor.fit(tabular_data)
                # sys.stdout.flush()
                # tab_embeds = predictor.extract_embedding(tabular_data)
                ######*******HALF NAS*******######

                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(144, pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, outputs)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_data[tabular_target]], axis=1)
                final_stacker_target = final_stacker[tabular_target]
                final_stacker_data = final_stacker.drop([tabular_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:

                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth)

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                    if downstream_model == 'XGBoostClassifier':
                        try:
                            print('XGBoostClassifier as DTM')

                            xgboost_max_depth = config['xgboost_max_depth']

                            xgboost_num_boost_round = config['xgboost_num_boost_round']
                            print(xgboost_num_boost_round)

                            grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}

                            fine_tuning_model = XGBClassifier()
                            fine_tuning_model.set_params(**grid)

                            seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                              "feature_processor": 'AutoMlFeatureGenerator',
                                                              "xgboost_max_depth": xgboost_max_depth,
                                                              "xgboost_num_boost_round": xgboost_num_boost_round,
                                                              "downstream_task": config['downstream_task']
                                                              }}

                            # Train the downstream model
                            fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                            valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                            # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                            downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                            if not os.path.exists(downstream_model_path_seed):
                                os.makedirs(downstream_model_path_seed)
                            downstream_model_path_seed_dir = downstream_model_path_seed

                            # Write the data to the JSON file
                            with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                      "w") as json_file:
                                json.dump(seed_config_dict_ds, json_file, indent=4)

                            ####Save the models here############

                            fine_tuning_model.save_model(
                                downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                            score = accuracy_score(valid_predictions, y_test)
                            # dummy_score = random.uniform(0,1)

                            objective_value = score
                            print(f'The computed objective value is {objective_value}')
                        except KeyError:
                            pass

                    if downstream_model in list(ensemble_clf_models.keys()):
                        # fine_tuning_model_stacked = None
                        tuple_list = []
                        for key, value in ensemble_clf_models.items():
                            if downstream_model == key:
                                for model_tuple in value.estimators:
                                    if model_tuple[0] == 'LGBClassifier':
                                        num_leaves = config['lgbm_num_leaves']
                                        max_depth = config['lgbm_max_depth']
                                        model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth)
                                        print('Hyperparam Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'XGBoostClassifier':
                                        # max_depth = config['lgbm_max_depth']
                                        try:
                                            max_depth = config['lgbm_max_depth']
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'CatBoostClassifier':
                                        # max_depth = config['lgbm_max_depth']
                                        try:
                                            max_depth = config['lgbm_max_depth']
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        ##cat boost has slightly wierd objectives. we can correct those here
                                        ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'KNNClassifier':
                                        n_neighbors = config['knn_n_neighbors']
                                        model_tuple[1].set_params(n_neighbors=n_neighbors)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "max_depth": max_depth,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
            if pretraining_model == 'Data2VecTextModel':
                pretraining_model = Data2VecTextModel.from_pretrained("facebook/flava-full")
                pretraining_feature_processor = AutoTokenizer.from_pretrained("facebook/flava-full")  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                tabular_target = user_defined_target  ## --- This will be a user argument

                # """ Text Processor"""
                ##Let us build the text dataset extractor -- IP DATA = test_df

                raw_text_data = text_data_extractor(test_df.data, text_column_list, single_frame_flag=True)
                tabular_data = tabular_extractor(test_df.data, text_column_list)
                print(f'The shape of the text data is {raw_text_data.shape}')
                print(f'The shape of the tabular data is {tabular_data.shape}')

                # text_data = raw_text_data.tolist()

                # Truncating our text sequences
                text_data = []
                truncate = lambda x: x[:512]
                for idx in range(len(raw_text_data)):
                    text_data.append(truncate(raw_text_data[idx]))

                print('Tokenizing Text..\n')
                sys.stdout.flush()

                inputs = []
                device = 'cuda'
                for text in text_data:
                    inputs.append(pretraining_feature_processor(text, return_tensors="pt").to(device))

                ## Token Processing (for token sequences greater than 512) : We need to process these sequences somehow
                new_inputs = []
                for i in range(len(inputs)):
                    if inputs[i]['input_ids'].size()[1] > 512:
                        print(f'The idx {i} has > 512 len')
                        tensor_list, token_type_list, attention_mask_list = [], [], []
                        for tidx in range(511):
                            tensor_list.append(inputs[i]['input_ids'][0][tidx])
                            token_type_list.append(inputs[i]['token_type_ids'][0][tidx])
                            attention_mask_list.append(inputs[i]['attention_mask'][0][tidx])

                            tensor = torch.cuda.IntTensor(tensor_list)
                            token_type_ids = torch.IntTensor(token_type_list)
                            attention_mask = torch.IntTensor(attention_mask_list)

                        inputs[i]['input_ids'] = torch.unsqueeze(tensor, 0)
                        inputs[i]['token_type_ids'] = torch.unsqueeze(token_type_ids, 0)
                        inputs[i]['attention_mask'] = torch.unsqueeze(attention_mask, 0)
                        new_inputs.append(inputs[i])

                outputs = []
                for i in range(len(inputs)):
                    print(f'ITR {i}')
                    sys.stdout.flush()
                    model_output_embedding = pretraining_model(**inputs[i].to(device)).last_hidden_state
                    ##Linear Layer :
                    Linear_Layer = nn.Linear(model_output_embedding.size()[2], pretraining_linear_hidden_size)
                    mm_embeddings = Linear_Layer(model_output_embedding)
                    # here we get the text and image embeddings
                    outputs.append(mm_embeddings.detach().numpy().flatten())  # Flatten the multimodal

                print('Done Tokenizing... \n')
                sys.stdout.flush()
                # import torch
                # torch.cuda.get_arch_list()
                print('Getting Tabular Embeddings \n')
                sys.stdout.flush()

                ######*******HALF NAS*******######
                # from autogluon.multimodal import MultiModalPredictor
                #
                # predictor = MultiModalPredictor(label='fraudulent')
                # predictor.fit(tabular_data)
                # sys.stdout.flush()
                # tab_embeds = predictor.extract_embedding(tabular_data)
                ######*******HALF NAS*******######

                tab_embeds = [torch.Tensor(tabx) for tabx in tabular_embeds]

                tab_linear_layer = nn.Linear(144, pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, outputs)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_data[tabular_target]], axis=1)
                final_stacker_target = final_stacker[tabular_target]
                final_stacker_data = final_stacker.drop([tabular_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'classification' and user_defined_task in ['classification', 'VQA']:

                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostClassifier':
                        print('CatBoostClassifier as DTM')

                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostClassifier(iterations=catboost_iterations, depth=catboost_depth)

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                    if downstream_model == 'XGBoostClassifier':
                        try:
                            print('XGBoostClassifier as DTM')

                            xgboost_max_depth = config['xgboost_max_depth']

                            xgboost_num_boost_round = config['xgboost_num_boost_round']
                            print(xgboost_num_boost_round)

                            grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}

                            fine_tuning_model = XGBClassifier()
                            fine_tuning_model.set_params(**grid)

                            seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                              "feature_processor": 'AutoMlFeatureGenerator',
                                                              "xgboost_max_depth": xgboost_max_depth,
                                                              "xgboost_num_boost_round": xgboost_num_boost_round,
                                                              "downstream_task": config['downstream_task']
                                                              }}

                            # Train the downstream model
                            fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                            valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                            # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                            downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                            if not os.path.exists(downstream_model_path_seed):
                                os.makedirs(downstream_model_path_seed)
                            downstream_model_path_seed_dir = downstream_model_path_seed

                            # Write the data to the JSON file
                            with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                      "w") as json_file:
                                json.dump(seed_config_dict_ds, json_file, indent=4)

                            ####Save the models here############

                            fine_tuning_model.save_model(
                                downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                            score = accuracy_score(valid_predictions, y_test)
                            # dummy_score = random.uniform(0,1)

                            objective_value = score
                            print(f'The computed objective value is {objective_value}')
                        except KeyError:
                            pass

                    if downstream_model in list(ensemble_clf_models.keys()):
                        # fine_tuning_model_stacked = None
                        tuple_list = []
                        for key, value in ensemble_clf_models.items():
                            if downstream_model == key:
                                for model_tuple in value.estimators:
                                    if model_tuple[0] == 'LGBClassifier':
                                        num_leaves = config['lgbm_num_leaves']
                                        max_depth = config['lgbm_max_depth']
                                        model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth)
                                        print('Hyperparam Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'XGBoostClassifier':
                                        # max_depth = config['lgbm_max_depth']
                                        try:
                                            max_depth = config['lgbm_max_depth']
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'CatBoostClassifier':
                                        # max_depth = config['lgbm_max_depth']
                                        try:
                                            max_depth = config['lgbm_max_depth']
                                        except KeyError:
                                            max_depth = 5
                                            continue
                                        ##cat boost has slightly wierd objectives. we can correct those here
                                        ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                        model_tuple[1].set_params(max_depth=max_depth)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value
                                    elif model_tuple[0] == 'KNNClassifier':
                                        n_neighbors = config['knn_n_neighbors']
                                        model_tuple[1].set_params(n_neighbors=n_neighbors)
                                        print('Hyperparameter Set')
                                        fine_tuning_model_stacked = value

                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "max_depth": max_depth,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = accuracy_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
        if user_defined_task == 'regression' and pretraining_task == 'regression':
            print('\n ...Automatically synthesizing end-to-end Deep ML pipeline for the regression task....\n')
            sys.stdout.flush()
            print(f'Enjoy your coffee {coffee} {coffee} ..\n ')
            sys.stdout.flush()
            print(f'Sampled Downstream Model is {downstream_model}')
            sys.stdout.flush()
            if pretraining_model == 'FlavaTextModel':  ##extension for Albef is to be added
                pretraining_model = FlavaTextModel.from_pretrained("facebook/flava-full")
                pretraining_feature_processor = AutoTokenizer.from_pretrained("facebook/flava-full")  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                tabular_target = user_defined_target  ## --- This will be a user argument

                # """ Text Processor"""
                ##Let us build the text dataset extractor -- IP DATA = test_df

                raw_text_data = text_data_extractor(test_df.data, text_column_list, single_frame_flag=True)
                tabular_data = tabular_extractor(test_df.data, text_column_list)
                print(f'The shape of the text data is {raw_text_data.shape}')
                print(f'The shape of the tabular data is {tabular_data.shape}')

                # text_data = raw_text_data.tolist()

                # Truncating our text sequences
                text_data = []
                truncate = lambda x: x[:512]
                for idx in range(len(raw_text_data)):
                    text_data.append(truncate(raw_text_data[idx]))

                print('Tokenizing Text..\n')
                sys.stdout.flush()

                inputs = []
                device = 'cuda'
                for text in text_data:
                    inputs.append(pretraining_feature_processor(text, return_tensors="pt").to(device))

                ## Token Processing (for token sequences greater than 512) : We need to process these sequences somehow
                new_inputs = []
                for i in range(len(inputs)):
                    if inputs[i]['input_ids'].size()[1] > 512:
                        print(f'The idx {i} has > 512 len')
                        tensor_list, token_type_list, attention_mask_list = [], [], []
                        for tidx in range(511):
                            tensor_list.append(inputs[i]['input_ids'][0][tidx])
                            token_type_list.append(inputs[i]['token_type_ids'][0][tidx])
                            attention_mask_list.append(inputs[i]['attention_mask'][0][tidx])

                            tensor = torch.cuda.IntTensor(tensor_list)
                            token_type_ids = torch.IntTensor(token_type_list)
                            attention_mask = torch.IntTensor(attention_mask_list)

                        inputs[i]['input_ids'] = torch.unsqueeze(tensor, 0)
                        inputs[i]['token_type_ids'] = torch.unsqueeze(token_type_ids, 0)
                        inputs[i]['attention_mask'] = torch.unsqueeze(attention_mask, 0)
                        new_inputs.append(inputs[i])

                outputs = []
                for i in range(len(inputs)):
                    print(f'ITR {i}')
                    sys.stdout.flush()
                    model_output_embedding = pretraining_model(**inputs[i].to(device)).last_hidden_state
                    ##Linear Layer :
                    Linear_Layer = nn.Linear(model_output_embedding.size()[2], pretraining_linear_hidden_size)
                    mm_embeddings = Linear_Layer(model_output_embedding)
                    # here we get the text and image embeddings
                    outputs.append(mm_embeddings.detach().numpy().flatten())  # Flatten the multimodal

                print('Done Tokenizing... \n')
                sys.stdout.flush()
                # import torch
                # torch.cuda.get_arch_list()
                print('Getting Tabular Embeddings \n')
                sys.stdout.flush()

                ######*******HALF NAS*******######
                # from autogluon.multimodal import MultiModalPredictor
                #
                # predictor = MultiModalPredictor(label='fraudulent')
                # predictor.fit(tabular_data)
                # sys.stdout.flush()
                # tab_embeds = predictor.extract_embedding(tabular_data)
                ######*******HALF NAS*******######

                tab_embeds = np.random.randn(50, 144)

                tab_embeds = [torch.Tensor(tabx) for tabx in tab_embeds]

                tab_linear_layer = nn.Linear(144, pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, outputs)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_data[tabular_target]], axis=1)
                final_stacker_target = final_stacker[tabular_target]
                final_stacker_data = final_stacker.drop([tabular_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Setstacked_ensemble_LGB_L1_2')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    problem = 'regression'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')
            if pretraining_model == 'Data2VecTextModel':
                pretraining_model = Data2VecTextModel.from_pretrained("facebook/flava-full")
                pretraining_feature_processor = AutoTokenizer.from_pretrained("facebook/flava-full")  #

                # fetch the hyperparam from the config space
                pretraining_num_layers = config['pretraining_num_layers']
                pretraining_hidden_size = config['pretraining_hidden_size']
                pretraining_linear_hidden_size = config['pretraining_linear_hidden_size']
                pretraining_pooling_kernel = config['pretraining_pooling_kernel']

                # set the value of the corresponding hyperparam to the config file of the ptm
                pretraining_model.config.num_layers = pretraining_num_layers
                pretraining_model.config.hidden_size = pretraining_hidden_size

                # We need to maintain the consistency for the hidden sizes across all the models
                pretraining_model.config.text_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.image_config.hidden_size = pretraining_hidden_size
                pretraining_model.config.multimodal_config.hidden_size = pretraining_hidden_size
                ##############################

                # here you can play with the hyperparams of the feature processor

                ############################
                # instantiate the PTM using the fetched config files for the selected hyperparams
                pretraining_model = FlavaForPreTraining(pretraining_model.config)
                pretraining_feature_processor = pretraining_feature_processor

                tabular_target = user_defined_target  ## --- This will be a user argument

                # """ Text Processor"""
                ##Let us build the text dataset extractor -- IP DATA = test_df

                raw_text_data = text_data_extractor(test_df.data, text_column_list, single_frame_flag=True)
                tabular_data = tabular_extractor(test_df.data, text_column_list)
                print(f'The shape of the text data is {raw_text_data.shape}')
                print(f'The shape of the tabular data is {tabular_data.shape}')

                # text_data = raw_text_data.tolist()

                # Truncating our text sequences
                text_data = []
                truncate = lambda x: x[:512]
                for idx in range(len(raw_text_data)):
                    text_data.append(truncate(raw_text_data[idx]))

                print('Tokenizing Text..\n')
                sys.stdout.flush()

                inputs = []
                device = 'cuda'
                for text in text_data:
                    inputs.append(pretraining_feature_processor(text, return_tensors="pt").to(device))

                ## Token Processing (for token sequences greater than 512) : We need to process these sequences somehow
                new_inputs = []
                for i in range(len(inputs)):
                    if inputs[i]['input_ids'].size()[1] > 512:
                        print(f'The idx {i} has > 512 len')
                        tensor_list, token_type_list, attention_mask_list = [], [], []
                        for tidx in range(511):
                            tensor_list.append(inputs[i]['input_ids'][0][tidx])
                            token_type_list.append(inputs[i]['token_type_ids'][0][tidx])
                            attention_mask_list.append(inputs[i]['attention_mask'][0][tidx])

                            tensor = torch.cuda.IntTensor(tensor_list)
                            token_type_ids = torch.IntTensor(token_type_list)
                            attention_mask = torch.IntTensor(attention_mask_list)

                        inputs[i]['input_ids'] = torch.unsqueeze(tensor, 0)
                        inputs[i]['token_type_ids'] = torch.unsqueeze(token_type_ids, 0)
                        inputs[i]['attention_mask'] = torch.unsqueeze(attention_mask, 0)
                        new_inputs.append(inputs[i])

                outputs = []
                for i in range(len(inputs)):
                    print(f'ITR {i}')
                    sys.stdout.flush()
                    model_output_embedding = pretraining_model(**inputs[i].to(device)).last_hidden_state
                    ##Linear Layer :
                    Linear_Layer = nn.Linear(model_output_embedding.size()[2], pretraining_linear_hidden_size)
                    mm_embeddings = Linear_Layer(model_output_embedding)
                    # here we get the text and image embeddings
                    outputs.append(mm_embeddings.detach().numpy().flatten())  # Flatten the multimodal

                print('Done Tokenizing... \n')
                sys.stdout.flush()
                # import torch
                # torch.cuda.get_arch_list()
                print('Getting Tabular Embeddings \n')
                sys.stdout.flush()

                ######*******HALF NAS*******######
                # from autogluon.multimodal import MultiModalPredictor
                #
                # predictor = MultiModalPredictor(label='fraudulent')
                # predictor.fit(tabular_data)
                # sys.stdout.flush()
                # tab_embeds = predictor.extract_embedding(tabular_data)
                ######*******HALF NAS*******######

                tab_embeds = np.random.randn(50, 144)

                tab_embeds = [torch.Tensor(tabx) for tabx in tab_embeds]

                tab_linear_layer = nn.Linear(144, pretraining_linear_hidden_size)
                tab_embeds = [tab_linear_layer(tab_embeds_idx) for tab_embeds_idx in tab_embeds]
                tab_embeds = [embdxxx.detach().numpy().flatten() for embdxxx in tab_embeds]

                ##Late Fusion of the linear, flattened embeddings
                final_embedding_concat = [np.append(text, tabular) for text, tabular in
                                          zip(tab_embeds, outputs)]
                embedding_ = np.array(final_embedding_concat, dtype=object)

                tensor_embedding = [torch.Tensor(emx) for emx in embedding_]
                max_embedding_len = max([len(x) for x in tensor_embedding])
                # Instead of pooling, we will apply a linear layer, which shall have an active hyperparameter

                max_pool = nn.MaxPool1d(2, stride=pretraining_pooling_kernel)
                pool_embedding = [max_pool(exx.unsqueeze(0)).squeeze(0).numpy() for exx in tensor_embedding]
                pooled_embeddings_final = np.array(pool_embedding, dtype=object)

                # Calculating the max sequence length after max pooling
                len_ = [arr.shape[0] for arr in pooled_embeddings_final]
                max_len = max(len_)
                print(f'The max len is {max_len}')
                sys.stdout.flush()

                # Now we will create a stacking model to stack the embeddings using a shallow stacker.
                lf_mm_embeddings = pd.DataFrame(pooled_embeddings_final, columns=['embeddings'])

                # Creating a feature dataframe for the obtained embedddings
                embedding_dataframe = pd.concat(
                    [lf_mm_embeddings[col].apply(pd.Series) for col in lf_mm_embeddings.columns],
                    axis=1,
                    ignore_index=True)

                final_stacker = pd.concat([embedding_dataframe, tabular_data[tabular_target]], axis=1)
                final_stacker_target = final_stacker[tabular_target]
                final_stacker_data = final_stacker.drop([tabular_target], axis=1)

                ##Before we fit the model with the lf_mm_embeddings, we need to perform a hold out
                X_train, X_test, y_train, y_test = train_test_split(final_stacker_data, final_stacker_target,
                                                                    test_size=0.3, random_state=42)

                ##Feature processing step:
                X_train = pd.DataFrame(X_train)
                X_test = pd.DataFrame(X_test)

                y_train = pd.DataFrame(y_train)
                y_test = pd.DataFrame(y_test)

                X_train_tranformed = downstream_feature_processor.fit_transform(X=X_train, y=y_train)
                #             X_train_tranformed = X_train_tranformed.interpolate()

                X_test_transformed = downstream_feature_processor.transform(X_test)
                #             X_test_transformed = X_test_transformed.interpolate()

                print(type(X_train_tranformed))
                if downstream_task == 'regression' and user_defined_task in ['regression', 'ITM']:
                    # Train the fine-tuning model
                    if downstream_model == 'CatBoostRegressor':
                        print('CatBoostRegressor as DTM')
                        catboost_iterations = config['catboost_iterations']
                        catboost_depth = config['catboost_depth']

                        fine_tuning_model = CatBoostRegressor(iterations=catboost_iterations, depth=catboost_depth)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "catboost_iterations": config['catboost_iterations'],
                                                          "catboost_depth": config['catboost_depth'],
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_transformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)

                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')

                if downstream_model == 'XGBoostRegressor':
                    try:
                        print('XGBoostRegressor as DTM')
                        xgboost_max_depth = config['xgboost_max_depth']
                        xgboost_num_boost_round = config['xgboost_num_boost_round']
                        print(xgboost_num_boost_round)
                        grid = {'num_boost': xgboost_num_boost_round, 'max_depth': xgboost_max_depth}
                        fine_tuning_model = XGBRegressor()
                        fine_tuning_model.set_params(**grid)
                        seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                          "feature_processor": 'AutoMlFeatureGenerator',
                                                          "xgboost_max_depth": xgboost_max_depth,
                                                          "xgboost_num_boost_round": xgboost_num_boost_round,
                                                          "downstream_task": config['downstream_task']
                                                          }}
                        # Train the downstream model
                        fine_tuning_model.fit(X_train_tranformed, y_train)  # set the eval metric here
                        valid_predictions = fine_tuning_model.predict(X_test_tranformed)
                        # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]
                        downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                        if not os.path.exists(downstream_model_path_seed):
                            os.makedirs(downstream_model_path_seed)
                        downstream_model_path_seed_dir = downstream_model_path_seed

                        # Write the data to the JSON file
                        with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json',
                                  "w") as json_file:
                            json.dump(seed_config_dict_ds, json_file, indent=4)
                        ####Save the models here############

                        fine_tuning_model.save_model(
                            downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                        score = r2_score(valid_predictions, y_test)
                        # dummy_score = random.uniform(0,1)

                        objective_value = score
                        print(f'The computed objective value is {objective_value}')
                    except KeyError:
                        pass
                if downstream_model in list(ensemble_reg_models.keys()):
                    # fine_tuning_model_stacked = None
                    tuple_list = []
                    for key, value in ensemble_reg_models.items():
                        if downstream_model == key:
                            for model_tuple in value.estimators:
                                if model_tuple[0] == 'LGBRegressor':
                                    problem = 'regression'
                                    num_leaves = config['lgbm_num_leaves']
                                    max_depth = config['lgbm_max_depth']
                                    model_tuple[1].set_params(num_leaves=num_leaves, max_depth=max_depth,
                                                              objective=problem)
                                    print('Hyperparam Setstacked_ensemble_LGB_L1_2')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'XGBoostRegressor':
                                    problem = 'regression'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    model_tuple[1].set_params(max_depth=max_depth, objective=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value
                                elif model_tuple[0] == 'CatBoostClassifier':
                                    problem = 'RMSE'
                                    # max_depth = config['lgbm_max_depth']
                                    try:
                                        max_depth = config['lgbm_max_depth']
                                    except KeyError:
                                        max_depth = 5
                                        continue
                                    ##cat boost has slightly wierd objectives. we can correct those here
                                    ##objective == 'multiclass' then objective == 'MultiClass' -- like this
                                    model_tuple[1].set_params(max_depth=max_depth, loss_function=problem)
                                    print('Hyperparameter Set')
                                    fine_tuning_model_stacked = value

                    seed_config_dict_ds = {"config": {"downstream_model": config['downstream_model'],
                                                      "feature_processor": 'AutoMlFeatureGenerator',
                                                      "max_depth": max_depth,
                                                      "downstream_task": config['downstream_task']
                                                      }}
                    # Train the downstream model
                    fine_tuning_model_stacked.fit(X_train_tranformed, y_train)  # set the eval metric here
                    valid_predictions = fine_tuning_model_stacked.predict(X_test_transformed)
                    # valid_predictions = [math.ceil(float_pred) for float_pred in valid_predictions]

                    downstream_model_path_seed = downstream_model_path + str(seed) + "/"

                    if not os.path.exists(downstream_model_path_seed):
                        os.makedirs(downstream_model_path_seed)
                    downstream_model_path_seed_dir = downstream_model_path_seed

                    # Write the data to the JSON file
                    with open(downstream_model_path_seed_dir + 'fine_tuning_model_config.json', "w") as json_file:
                        json.dump(seed_config_dict_ds, json_file, indent=4)

                    ####Save the models here############

                    fine_tuning_model.save_model(
                        downstream_model_path_seed_dir + "fine_tuned_model" + str(seed) + "_.json")

                    score = r2_score(valid_predictions, y_test)
                    # dummy_score = random.uniform(0,1)

                    objective_value = score
                    print(f'The computed objective value is {objective_value}')
    return objective_value  # + add the valse scores


## Designing Config Space - #####******RESTUDY THE CONFIG HYERPARAMS----
cs = ConfigurationSpace()
# Define hyperparameters for pretraining models
pretraining_model = CategoricalHyperparameter('pretraining_model', choices=pretraining_model_choices,
                                              default_value='FlavaFeatureProcessor')
pretraining_feature_processor = CategoricalHyperparameter('pretraining_feature_processor',
                                                          choices=['FlavaProcessor', 'AlbefProcessor'],
                                                          default_value='FlavaProcessor')
pretraining_num_layers = UniformIntegerHyperparameter('pretraining_num_layers', lower=1, upper=10, default_value=None)
pretraining_hidden_size = UniformIntegerHyperparameter('pretraining_hidden_size', lower=108, upper=288,
                                                       default_value=108, q=12)
pretraining_linear_hidden_size = UniformIntegerHyperparameter('pretraining_linear_hidden_size', lower=108, upper=360,
                                                              default_value=108)
pretraining_pooling_kernel = UniformIntegerHyperparameter('pretraining_pooling_kernel', lower=2, upper=4,
                                                          default_value=2)

pretraining_task = CategoricalHyperparameter('pretraining_task', choices=['classification', 'regression', 'ITM', 'VQA'],
                                             default_value='classification')
downstream_task = CategoricalHyperparameter('downstream_task', choices=['classification', 'regression'],
                                            default_value='classification')

downstream_model = CategoricalHyperparameter('downstream_model', choices=downstrem_choices, default_value=None)
# Define hyperparameters for CatBoost
catboost_iterations = UniformIntegerHyperparameter('catboost_iterations', lower=100, upper=500, default_value=150)
catboost_depth = UniformIntegerHyperparameter('catboost_depth', lower=3, upper=10, default_value=5)

# Define hyperparameters for XGBoost
xgboost_num_boost_round = UniformIntegerHyperparameter('xgboost_num_boost_round', lower=100, upper=500,
                                                       default_value=150)
xgboost_max_depth = UniformIntegerHyperparameter('xgboost_max_depth', lower=3, upper=10, default_value=5)

# Define hyperparameters for LGBMBoost
lgbm_num_leaves = UniformIntegerHyperparameter('lgbm_num_leaves', lower=50, upper=300, default_value=128)
lgbm_max_depth = UniformIntegerHyperparameter('lgbm_max_depth', lower=3, upper=10, default_value=5)

# Add hyperparameters to pretraining configuration space
cs.add_hyperparameters([pretraining_model,
                        pretraining_num_layers,
                        pretraining_hidden_size,
                        pretraining_feature_processor,
                        pretraining_linear_hidden_size,
                        pretraining_task,
                        downstream_task,
                        pretraining_pooling_kernel,
                        downstream_model,
                        catboost_iterations,
                        catboost_depth,
                        xgboost_num_boost_round,
                        xgboost_max_depth,
                        lgbm_num_leaves,
                        lgbm_max_depth])

# Define conditions for pretraining configuration space
condition_ptm_num_layers = InCondition(pretraining_num_layers, pretraining_model, pretraining_model_choices)
condition_ptm_hidden_size = InCondition(pretraining_hidden_size, pretraining_model, pretraining_model_choices)
condition_ptm_linear_hidden_size = InCondition(pretraining_linear_hidden_size, pretraining_model,
                                               pretraining_model_choices)
condition_ptm_pooling_kernel = InCondition(pretraining_pooling_kernel, pretraining_model, pretraining_model_choices)

condition_ptm_processor_flava = InCondition(pretraining_feature_processor, pretraining_model,
                                            ['FlavaFeatureProcessor', 'FlavaVQA', 'FlavaITM'])
condition_ptm_processor_albef = InCondition(pretraining_feature_processor, pretraining_model,
                                            ['AlbefFeatureProcessor', 'AlbefVQA', 'AlbefITM'])

condition_processor_conjunction = AndConjunction(condition_ptm_processor_flava,
                                                 condition_ptm_processor_albef)

# Define conditions for downstream configuration space
catboost_depth_condition = InCondition(catboost_depth, downstream_model, ['CatBoostClassifier', 'CatBoostRegressor'])
catboost_itr_condition = InCondition(catboost_iterations, downstream_model, ['CatBoostClassifier', 'CatBoostRegressor'])
xgboost_depth_condition = InCondition(xgboost_max_depth, downstream_model, ['XGBoostClassifier', 'XGBoostRegressor'])
xgboost_itr_condition = InCondition(xgboost_num_boost_round, downstream_model,
                                    ['XGBoostClassifier', 'XGBoostRegressor'])

# LGBM conditions
lgb_num_leaves_condition = InCondition(lgbm_num_leaves, downstream_model, models_consisting_lgb)
lgbm_max_depth_condition = InCondition(lgbm_max_depth, downstream_model, models_consisting_lgb)

# Add conditions to the configuration space
# Define conditions for pretraining configuration space
# condition_ptm_task_regression = ForbiddenEqualsClause(pretraining_task, 'regression')
# condition_ptm_task_ITM = ForbiddenEqualsClause(pretraining_task, 'ITM')

# Define conditions for downstream configuration space
condition_downstream_task_regression = InCondition(downstream_task,pretraining_task,['classification', 'VQA',
                                                                                     'regression', 'ITM'])
# condition_downstream_task_ITM = InCondition(pretraining_task,downstream_task , ['classification'])

# PTM conditions
cs.add_conditions([condition_ptm_num_layers,
                   condition_ptm_hidden_size,
                   condition_ptm_linear_hidden_size,
                   condition_processor_conjunction,
                   condition_ptm_pooling_kernel,
                   # DTM conditions
                   catboost_depth_condition,
                   catboost_itr_condition,
                   xgboost_depth_condition,
                   xgboost_itr_condition,
                   lgb_num_leaves_condition,
                   lgbm_max_depth_condition,
                   condition_downstream_task_regression])

# Add forbidden clauses for the sampling of the configurations

downstream_classification_task_and_regression_models = ForbiddenAndConjunction(
    ForbiddenEqualsClause(downstream_task, "classification"),
    ForbiddenInClause(downstream_model, regression_model_list
                      ))

downstream_regression_task_and_classification_models = ForbiddenAndConjunction(
    ForbiddenEqualsClause(downstream_task, "regression"),
    ForbiddenInClause(downstream_model, classification_model_list))

pretraining_classification_task_and_regression_models = ForbiddenAndConjunction(
    ForbiddenInClause(pretraining_task, ["classification","VQA"]),
    ForbiddenInClause(downstream_model, regression_model_list))

pretraining_regression_task_and_classification_models = ForbiddenAndConjunction(
    ForbiddenInClause(pretraining_task, ["regression", "ITM"]),
    ForbiddenInClause(downstream_model, classification_model_list))

pretraining_VQA_task_and_regression_models = ForbiddenAndConjunction(
    ForbiddenInClause(pretraining_model, ['FlavaVQA', 'AlbefVQA']),
    ForbiddenEqualsClause(downstream_task, "regression"), ##new line
    ForbiddenInClause(downstream_model, regression_model_list))

pretraining_ITM_task_and_classification_models = ForbiddenAndConjunction(
    ForbiddenEqualsClause(pretraining_task, "ITM"),
    ForbiddenEqualsClause(downstream_task, "classification"), ##new line
    ForbiddenInClause(downstream_model, classification_model_list))



cs.add_forbidden_clauses([downstream_classification_task_and_regression_models,
                          downstream_regression_task_and_classification_models,
                          pretraining_classification_task_and_regression_models,
                          pretraining_regression_task_and_classification_models])
#                           pretraining_VQA_task_and_regression_models,
#                           pretraining_ITM_task_and_classification_models,
#                           condition_ptm_task_regression,
#                           condition_ptm_task_ITM])
cs.sample_configuration()

## These are all active hyperparameters in my configurtion space
cs.get_active_hyperparameters(cs.sample_configuration())

# Dummy Meta Dataset
meta_dataset = {
    'data1': {
        'config': {
            'pretraining_task': 'classification',
            'downstream_task': 'classification',
            'pretraining_linear_hidden_size': 256,
            'pretraining_model': 'FlavaFeatureProcessor',
            'pretraining_feature_processor': 'FlavaProcessor',
            'pretraining_num_layers': 2,
            'pretraining_hidden_size': 108,
            'pretraining_pooling_kernel': 2,
            'downstream_model': 'weighted_ensemble_CAT_L1_1',
            'lgbm_max_depth': 4,
            'lgbm_num_leaves': 38
        },
        'score': 0.9

    },
    'data2': {
        'config': {
            'pretraining_task': 'classification',
            'downstream_task': 'classification',
            'pretraining_linear_hidden_size': 256,
            'pretraining_model': 'FlavaFeatureProcessor',
            'pretraining_feature_processor': 'FlavaProcessor',
            'pretraining_num_layers': 2,
            'pretraining_hidden_size': 144,
            'pretraining_pooling_kernel': 2,
            'downstream_model': 'CatBoostClassifier',
            'catboost_iterations': 50,
            'catboost_depth': 3

        },
        'score': 0.8

    },
    'data3': {
        'config': {
            'pretraining_task': 'classification',
            'downstream_task': 'classification',
            'pretraining_linear_hidden_size': 256,
            'pretraining_model': 'FlavaFeatureProcessor',
            'pretraining_feature_processor': 'FlavaProcessor',
            'pretraining_num_layers': 2,
            'pretraining_hidden_size': 288,
            'pretraining_pooling_kernel': 4,
            'downstream_model': 'stacked_ensemble_LGB_L1_2',
            'lgbm_max_depth': 4,
            'lgbm_num_leaves': 38
        },
        'score': 0.65
    }
}

initial_design = []
run_history = RunHistory()
ID = 3
for key, data in meta_dataset.items():
    config = Configuration(cs, values=data['config'], allow_inactive_with_values=True)
    initial_design.append(config)
    print(meta_dataset[key]['score'])
    id_ = np.random.randint(10, 100)
    # Add the configurations to the run history
    run_history.add(config=config, instance=id_, seed=seed, cost=meta_dataset[key]['score'], budget=120)
    ID += 1

for k in run_history:
    print(k)

##You should pass the pipeline configuration extracted from the metadataset
time_limit = 1500
path = '/mnt/server-home/TUE/20210962/smac3_output/runs/' + str(seed) + "/"

if not os.path.exists(path):
    os.makedirs(path)
output_dir = '/mnt/server-home/TUE/20210962/smac3_output/runs/' + str(seed) + "/"

scenario = Scenario(
    configspace=cs,
    objectives='score',  # optimize quality metric #or try cost
    walltime_limit=time_limit,  # set the time limit in seconds
    seed=seed,
    output_directory=output_dir

)

# In[27]:
# In[28]:


cs.get_default_configuration()

# In[29]:


from smac.model.random_forest import RandomForest

rf = RandomForest(cs)
expected_improvement = EI()
smac = ACF(
    scenario=scenario,
    target_function=objective_function,
    acquisition_function=expected_improvement,
    model=rf)

# In[30]:


# incumbent  = smac.optimize() #calling the custom defined smac function


# In[31]:


import signal
import pickle
from contextlib import contextmanager


# Define a context manager to set the timeout for each optimization iteration
@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Execution timed out.")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)


# Define the number of iterations and timeout duration
num_iterations = 10
timeout_seconds = 1000  # Set the timeout duration to 100 seconds

# Define a variable to store the best incumbent configuration and its corresponding objective value
best_incumbent = None
best_objective_value = float('-inf')
incumbent_list = []

# Define a directory to store the evaluated models
model_directory = "/mnt/server-home/TUE/20210962/smac3_output/" + str(
    seed) + "evaluated_models/"

# Create the model directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Optimization loop
for i in range(num_iterations):
    try:
        print(f'Starting Optimization Incumbent ITR {i}')
        sys.stdout.flush()

        # with time_limit(timeout_seconds):
        incumbent = smac.optimize()
        sys.stdout.flush()

        # Get the objective value of the incumbent configuration
        objective_value = objective_function(incumbent, seed=i)

        # Update the best incumbent if necessary
        if objective_value > best_objective_value:
            best_incumbent = incumbent
            best_objective_value = objective_value

        # Print the current iteration's information
        print(f"Iteration {i + 1}: Best objective value = {best_objective_value}")
        sys.stdout.flush()


    except (TimeoutError, KeyboardInterrupt, KeyError):
        print(f"Iteration {i + 1} timed out.")
        sys.stdout.flush()

        continue

# smac.solver.runhistory.save_model(best_incumbent, output_directory)

# Print the final best incumbent and its objective value
print("Final Best Incumbent:")
sys.stdout.flush()

print(best_incumbent)
print("Objective Value:")
sys.stdout.flush()

path = '/mnt/server-home/TUE/20210962/smac3_output/runs/' + str(seed) + "/best_config/"

if not os.path.exists(path):
    os.makedirs(path)
output_dir = '/mnt/server-home/TUE/20210962/smac3_output/runs/' + str(seed) + "/best_config/"

# Save the best model
with open(output_dir + 'best_incumbent.pkl', 'wb') as f:
    pickle.dump(best_incumbent, f)

print(best_objective_value)
# In[ ]:


import pickle
# Load the best_incumbent pickle file
with open(output_dir + 'best_incumbent.pkl', 'rb') as f:
    best_incumbent = pickle.load(f)
# Run the best_incumbent
result = best_incumbent # Assuming best_incumbent is a callable object
