from typing import List

def read_text(path):
    with open(path,"r",encoding="utf-8") as f:
        return f.read()
        
def proportional_integer_allocation(weights: List[float], total_count: int) -> List[int]:
    """
    将 total_count 个资源按 weights 比例分配，并进行误差修正，使总和等于 total_count。

    Args:
        weights: 每项的权重（如每行字幕的长度）。
        total_count: 总资源数（如帧数）。
    
    Returns:
        List[int]: 每项被分配到的资源数。
    """
    if total_count <= 0 or not weights or sum(weights) == 0:
        return [0] * len(weights)
    
    total_weight = sum(weights)
    raw_allocations = [(w / total_weight) * total_count for w in weights]
    int_allocations = [int(x) for x in raw_allocations]
    remainders = [x - int_x for x, int_x in zip(raw_allocations, int_allocations)]

    # 剩余资源数量
    remaining = total_count - sum(int_allocations)

    # 将剩下的资源分配给最大余数对应的项
    remainder_indices = sorted(range(len(remainders)), key=lambda i: remainders[i], reverse=True)
    for i in range(remaining):
        int_allocations[remainder_indices[i]] += 1

    return int_allocations