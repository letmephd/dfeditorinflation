import torch

def find_bounding_box(mask):
    """
    找到掩码中所有 True 值的最小边界框 (bounding box)。
    """
    coords = torch.nonzero(mask, as_tuple=False)
    y_min, x_min = torch.min(coords, dim=0)[0][2:]
    y_max, x_max = torch.max(coords, dim=0)[0][2:]
    return y_min.item(), x_min.item(), y_max.item(), x_max.item()

def move_and_union_masks(mask1, mask2):
    """
    将 mask2 移动到与 mask1 的 bounding box 对齐，并计算两者的并集。
    返回移动后的 mask2 以及并集 mask。
    """
    # 找到两个mask的最小bounding box
    y_min1, x_min1, y_max1, x_max1 = find_bounding_box(mask1)
    y_min2, x_min2, y_max2, x_max2 = find_bounding_box(mask2)

    # 计算两个bounding box的中心
    center1_y, center1_x = (y_min1 + y_max1) // 2, (x_min1 + x_max1) // 2
    center2_y, center2_x = (y_min2 + y_max2) // 2, (x_min2 + x_max2) // 2

    # 计算移动的偏移量
    dy, dx = center1_y - center2_y, center1_x - center2_x

    # 移动mask2
    mask2_moved = torch.roll(mask2, shifts=(dy, dx), dims=(0, 1))

    # 创建并集mask
    mask_union = mask1 | mask2_moved

    # 修正移出边界的情况
    y_start, y_end = max(0, dy), mask1.shape[0] - max(0, -dy)
    x_start, x_end = max(0, dx), mask1.shape[1] - max(0, -dx)
    mask_union[y_start:y_end, x_start:x_end] = mask1[y_start:y_end, x_start:x_end] | mask2_moved[max(0, -dy):mask1.shape[0] - max(0, dy), max(0, -dx):mask1.shape[1] - max(0, dx)]

    return mask2_moved, mask_union

def main(mask1, mask2):
    # 对第一个mask，将第二个mask进行移动，计算并集
    mask2_moved1, mask_union1 = move_and_union_masks(mask1, mask2)

    # 对第二个mask，将第一个mask进行移动，计算并集
    mask1_moved2, mask_union2 = move_and_union_masks(mask2, mask1)

    # 调整mask2_moved1和mask1_moved2，使得True的值个数相同
    num_true_1 = mask_union1.sum().item()
    num_true_2 = mask_union2.sum().item()
    
    if num_true_1 > num_true_2:
        difference = num_true_1 - num_true_2
        indices_to_flip = torch.nonzero(mask_union1)[:difference]
        mask_union1[indices_to_flip[:, 0], indices_to_flip[:, 1]] = 0
    elif num_true_2 > num_true_1:
        difference = num_true_2 - num_true_1
        indices_to_flip = torch.nonzero(mask_union2)[:difference]
        mask_union2[indices_to_flip[:, 0], indices_to_flip[:, 1]] = 0

    return mask_union1, mask_union2
