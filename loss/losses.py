from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from metrics.metrics import bbox_iou


def smooth_bce_constants(eps: float = 0.1) -> Tuple[float, float]:
    """Smooth Binary Cross Entropy.
    https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    :param eps: epsilon value for smoothing
    :return: positive, negative label smoothing BCE targets
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Binary Cross Entropy Blur Logit Loss with reduced missing label effects."""

    def __init__(self, alpha: float = 0.05) -> None:
        """Initialize instance.
        :param alpha: alpha parameter
        """
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute loss
        :param pred: predictions
        :param ground_truth: true labels
        :return: Loss value
        """
        loss = self.loss_fcn(pred, ground_truth)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - ground_truth  # reduce only missing label effects
        # dx = (pred - ground_truth).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss module.
    This class wraps focal loss around existing loss_fcn(),
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    """

    def __init__(self, loss_fcn: Callable, gamma: float = 1.5, alpha: float = 0.25) -> None:
        """Initialize focal loss.
        :param loss_fcn: loss function to wrap
        :param gamma: gamma parameter for focal loss
        :param alpha: alpha parameter for focal loss
        """
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction  # type: ignore
        # required to apply FL to each element
        self.loss_fcn.reduction = "none"  # type: ignore

    def forward(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        :param pred: predictions
        :param ground_truth: truth labels
        :return: focal loss value.
        """
        loss = self.loss_fcn(pred, ground_truth)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = ground_truth * pred_prob + (1 - ground_truth) * (1 - pred_prob)
        alpha_factor = ground_truth * self.alpha + (1 - ground_truth) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """QFocal loss module.

    This class wraps Quality focal loss around existing loss_fcn(),
    i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss, gamma=1.5)
    """

    def __init__(self, loss_fcn: Callable, gamma: float = 1.5, alpha: float = 0.25) -> None:
        """Initialize QFocal loss.
        :param loss_fcn: loss function to wrap
        :param gamma: gamma parameter for focal loss.
        :param alpha: alpha parameter for focal loss.
        """
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction  # type: ignore
        # required to apply FL to each element
        self.loss_fcn.reduction = "none"  # type: ignore

    def forward(self, pred: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute QFocal loss.
        :param pred: predictions
        :param ground_truth: truth labels
        :return: focal loss value.
        """
        loss = self.loss_fcn(pred, ground_truth)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = ground_truth * self.alpha + (1 - ground_truth) * (1 - self.alpha)
        modulating_factor = torch.abs(ground_truth - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Compute YOLO Loss."""

    def __init__(self, model: nn.Module, autobalance: bool = False) -> None:
        """Initialize instance
        :param model: YOLOModel or nn.Module which last layer is YOLOHead
        :param autobalance: Auto balance
        """
        self._yolo_layers = model.yolo_layers
        self._module_list = model.module_list
        self._anchor_t = model.hyp['anchor_t']
        self.sort_obj_iou = False
        device = next(model.parameters()).device  # get model device
        hyp: Dict[str, Any] = model.hyp  # type: ignore

        # Define criteria
        bce_cls: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp["cls_pw"]], device=device)
        )
        bce_obj: nn.Module = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp["obj_pw"]], device=device)
        )

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_bce_constants(
            eps=hyp.get("label_smoothing", 0.0)
        )  # positive, negative BCE targets

        # Focal loss
        gamma = hyp["fl_gamma"]  # focal loss gamma
        if gamma > 0:
            bce_cls, bce_obj = FocalLoss(bce_cls, gamma), FocalLoss(bce_obj, gamma)

        head = model.module_list[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(head.nl, [4.0, 1.0, 0.25, 0.06, 0.02])

        # stride 16 index
        self.ssi = list(head.stride).index(16) if autobalance else 0  # type: ignore
        self.bce_cls, self.bce_obj, self.gr, self.hyp, self.autobalance = (
            bce_cls,
            bce_obj,
            1.0,
            hyp,
            autobalance,
        )

        self.na: int = head.na  # type: ignore
        self.nc: int = head.nc  # type: ignore
        self.nl: int = head.nl  # type: ignore
        self.anchors: torch.Tensor = head.anchors  # type: ignore

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:  # predictions, targets, model
        """Compute YOLO Loss.
        :param preds: predictions
        :param targets: truth labels
        :return:
            Total loss (lbox + lobj + lcls) * batch_size,
            Each losses (lbox, lobj, lcls)
        """
        device = targets.device
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        tcls, tbox, indices, anchors = self.build_targets(preds, targets)  # targets
        target_obj = np.array([])
        # Losses
        for i, pred_i in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            target_obj = torch.zeros_like(pred_i[..., 0], device=device)  # target obj

            num_targets = b.shape[0]  # number of targets
            if num_targets:
                ps = pred_i[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression loss
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox.T, tbox[i], x1y1x2y2=False, c_iou=True
                )  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Object loss
                score_iou = iou.detach().clamp(0).type(target_obj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (
                        b[sort_id],
                        a[sort_id],
                        gj[sort_id],
                        gi[sort_id],
                        score_iou[sort_id],
                    )
                target_obj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification loss
                if self.nc > 1:  # cls loss (only if multiple classes)
                    _targets = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    _targets[range(num_targets), tcls[i]] = self.cp
                    lcls += self.bce_cls.to(device)(ps[:, 5:], _targets)  # BCE

            obji = self.bce_obj.to(device)(pred_i[..., 4], target_obj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = target_obj.shape[0]  # batch size
        loss = lbox + lobj + lcls

        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, preds: torch.Tensor, targets: torch.Tensor) \
            -> Tuple[
                List[torch.Tensor],
                List[torch.Tensor],
                List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
                List[torch.Tensor],
            ]:
        """Build targets for compute_loss(), input targets(image,class,x,y,w,h).
        :param preds: predictions
        :param targets: truth labels
        :returns:
            target classes of each layer
            target boxes of each layer
            target indices of each layer (batch, anchor, x, y)
            target anchor size of each layer
        """
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(6, device=targets.device)  # normalized to gridspace gain
        off = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float()  # overlap offsets

        g = 0.5  # offset
        for i, jj in enumerate(self._yolo_layers):
            # get number of grid points and anchor vec for this yolo layer
            anchors = self._module_list[jj].anchor_vec
            gain[2:] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            a, t, offsets = [], targets * gain, 0
            if nt:
                na = anchors.shape[0]  # number of anchors
                at = torch.arange(na).view(na, 1).repeat(1, nt)  # anchor tensor, same as .repeat_interleave(nt)
                r = t[None, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self._anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n) = wh_iou(anchors(3,2), gwh(n,2))
                a, t = at[j], t.repeat(na, 1, 1)[j]  # filter

                # overlaps
                gxy = t[:, 2:4]  # grid xy
                z = torch.zeros_like(gxy)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxy % 1. > (1 - g)) & (gxy < (gain[[2, 3]] - 1.))).T
                a, t = torch.cat((a, a[j], a[k], a[l], a[m]), 0), torch.cat((t, t[j], t[k], t[l], t[m]), 0)
                offsets = torch.cat((z, z[j] + off[0], z[k] + off[1], z[l] + off[2], z[m] + off[3]), 0) * g

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        return tcls, tbox, indices, anch
