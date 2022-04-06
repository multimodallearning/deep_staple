import torch

# 2d metrics
###
###

def dice2d(predicted_lbls: torch.tensor, target_lbls: torch.tensor, one_hot_torch_style: bool,
           nan_for_unlabeled_target=True) -> torch.tensor:
    r"""
    Calculates the dice score for a batch of 2d labels.
    Parameters:
        predicted_lbls (torch.tensor): The one-hot-encoded predicted labels to be assessed.
            Dimensions need to be (B,label_count,H,W).
            For alternate dimension see param 'one_hot_torch_style=True'.
        target_lbls (torch.tensor): The ground-truth which serves as a reference.
        one_hot_torch_style (bool): Label input can be (B,H,W,label_count) as returned by torch.nn.functional.one_hot()
        nan_for_unlabeled_target (bool): Yields 'nan' if target label and predicted label are both zero.
            Reduce with torch.nanmean() to exclude those in mean score.
    Returns:
        (torch.tensor): 2d Tensor (Bxlabel_count) containing the score per batch sample and class.
    .. _Optimizing the Dice Score and Jaccard Index for Medical Image Segmentation: Theory & Practice
        https://arxiv.org/abs/1911.01685
    """
    assert predicted_lbls.dim() == 4, \
        f"Dimensions of volume must be (B,label_count,H,W) or (B,H,W,label_count) if one_hot_torch_style=True' but is {predicted_lbls.shape}"
    assert predicted_lbls.size() == target_lbls.size(), \
        f"Dimensions of predicted_lbls and target_lbls must match but are predicted_lbls={predicted_lbls.shape}, target_lbls={target_lbls.shape}"

    return _diceNd(predicted_lbls, target_lbls, one_hot_torch_style, nan_for_unlabeled_target)



# 3d metrics
###
###

def dice3d(predicted_lbls: torch.tensor, target_lbls: torch.tensor, one_hot_torch_style: bool,
           nan_for_unlabeled_target=True) -> torch.tensor:
    r"""
        Calculates the dice score for a batch of 3d labels.
        Parameters:
            predicted_lbls (torch.tensor): The one-hot-encoded predicted labels to be assessed.
                Dimensions need to be (B,label_count,D,H,W).
                For alternate dimension see param 'one_hot_torch_style=True'.
            target_lbls (torch.tensor): The ground-truth which serves as a reference.
            one_hot_torch_style (bool): Label input can be (B,D,H,W,label_count) as returned by torch.nn.functional.one_hot()
            nan_for_unlabeled_target (bool): Yields 'nan' if target label and predicted label are both zero.
                Reduce with torch.nanmean() to exclude those in mean score.
        Returns:
            (torch.tensor): 2d Tensor (B,label_count) containing the score per batch sample and class.
        .. _Optimizing the Dice Score and Jaccard Index for Medical Image Segmentation: Theory & Practice
            https://arxiv.org/abs/1911.01685
        """

    assert predicted_lbls.dim() == 5, \
        f"Dimensions of volume must be (B,label_count,D,H,W) or (B,D,H,W,label_count) if 'one_hot_torch_style=True' but is {predicted_lbls.shape}"
    assert predicted_lbls.size() == target_lbls.size(), \
        f"Dimensions of predicted_lbls and target_lbls must match but are predicted_lbls={predicted_lbls.shape}, target_lbls={target_lbls.shape}"

    return _diceNd(predicted_lbls, target_lbls, one_hot_torch_style, nan_for_unlabeled_target)



# protected functions
#
#

def _diceNd(predicted_lbls: torch.tensor, target_lbls: torch.tensor, one_hot_torch_style,
           nan_for_unlabeled_target=True) -> torch.tensor:
    r"""Calculates the dice score for a batch of labels.
    Parameters:
        predicted_lbls (torch.tensor): The one-hot-encoded predicted_labels to be assessed.
            Dimensions need to be (B,label_count,[n-dims]).
        target_lbls (torch.tensor): The one-hot-encoded ground-truth
            which serves as a reference.
            one_hot_torch_style (bool): Label input can be (B,[n-dims],label_count) as returned by torch.nn.functional.one_hot()
            nan_for_unlabeled_target (bool): Yields 'nan' if target label and predicted label are both zero.
                Reduce with torch.nanmean() to exclude those in mean score.
    Returns:
        (torch.tensor): 2d Tensor (B,label_count) containing the score per batch sample and class.
    """

    if one_hot_torch_style:
        predicted_lbls = torch.movedim(predicted_lbls, -1, 1)
        target_lbls = torch.movedim(target_lbls, -1, 1)

    b_size, label_count, *_ = target_lbls.size()

    dice = torch.zeros((b_size , label_count))

    # Check every label seperately
    for label_num in range(0, label_count):
        all_pos_predicted = predicted_lbls[:,label_num] == 1
        all_pos_labeled = target_lbls[:,label_num] == 1 # Collect total count of labelled solution and add to sum (TP + FN)

        # Count true positives (if pixel has label label_num and the ref label-grid has label_num too)
        t_p = torch.logical_and(all_pos_predicted, all_pos_labeled)
        t_p = t_p.reshape(b_size, -1).sum(-1) # Then sum over last n-dimensions

        pos_predicted_count = all_pos_predicted.reshape(b_size, -1).sum(-1) # Collect total count of all positively predicted and add to sum (FP + TP)
        pos_labeled_count = all_pos_labeled.reshape(b_size, -1).sum(-1)

        if nan_for_unlabeled_target:
            nan_control = 0. # Will make divisioin nan if no pos labeled pixels are given for label and none are predicted
        else:
            nan_control = 1e-10

        dice[:,label_num] = (2.0*t_p/(pos_predicted_count + pos_labeled_count + nan_control)) # 2d Tensor (b_sizexlabel_count) i.e. scalar dice score per batch_sample per label of batch
        # dice = 2*TP over 2*TP+FP+FN === 2*TP over (TP+FP)+(TP+FN) === 2*TP over (pos_predicted)+(pos_labeled)

    return dice