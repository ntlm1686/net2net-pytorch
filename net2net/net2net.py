# Reference: Net2Net https://arxiv.org/pdf/1511.05641.pdf
import torch
import torch.nn as nn

from typing import List


def random_mapping_one(
    j: int,  # 1 <= j <= q, the number to be mapped
    n: int,  # 1 <= n <= q, the max number of target space
):
    """A random mapping function"""
    return j if j < n else torch.randint(n, (1,)).item()


def random_mapping_vector(
    l: List,
    n: int,
):
    return list(map(lambda x: random_mapping_one(x, n), l))


def wider_ln_(ln_teacher: nn.Module, ln_student: nn.Module, g: List[int]):
    """Net2Wider for LayerNorm, this function modifies inputs"""
    for j in range(len(g)):
        ln_student.weight.data[j] = ln_teacher.weight.data[g[j]]
        ln_student.bias.data[j] = ln_teacher.bias.data[g[j]]

def rmap(
    n: int,
    q: int,
):
    return random_mapping_vector(range(q), n)

def map_g(
    l1_teacher: nn.Module,
    l1_student: nn.Module,
) -> List[int]:
    teacher_hidden = l1_teacher.weight.shape[0]  # output units of teacher
    student_hidden = l1_student.weight.shape[0]  # output units of student
    return random_mapping_vector(range(student_hidden), teacher_hidden)


def wider_(
    l1_teacher: nn.Module,
    l2_teacher: nn.Module,
    l1_student: nn.Module,
    l2_student: nn.Module,
    g: List[int]
):
    """Net2WiderNet for two consecutive PyTorch Fully Connected Layer/ Conv1d layer, this function modifies inputs

    Args:
        teacher:
            l1_teacher: the first layer of teacher net
            l2_teacher: the second layer of teacher net
        student:
            l1_student: the first layer of student net
            l2_student: the second layer of student net
    """
    assert l1_teacher.weight.shape[0] == l2_teacher.weight.shape[1]
    assert l1_student.weight.shape[0] == l2_student.weight.shape[1]
    student_hidden = l1_student.weight.shape[0]
    for j in range(student_hidden):
        # replace rows
        l1_student.weight.data[j] = l1_teacher.weight.data[g[j]]
        l2_student.weight.data[:, j] = 1/g.count(g[j]) \
            * l2_teacher.weight.data[:, g[j]]
    if l1_student.bias is not None:
        if l1_teacher.bias is None:
            l1_student.bias.data.fill_(0)
        else:
            for j in range(student_hidden):
                l1_student.bias.data[j] = l1_teacher.bias.data[g[j]]
    if l2_student.bias is not None:
        if l2_teacher.bias is None:
            l2_student.bias.data.fill_(0)
        else:
            for j in range(student_hidden):
                l2_student.bias.data = l2_teacher.bias.data


def wider_first_(
    l1_teacher: nn.Module,
    l1_student: nn.Module,
    g: List[int]
) -> None:
    student_hidden = l1_student.weight.shape[0]
    for j in range(student_hidden):
        l1_student.weight.data[j] = l1_teacher.weight.data[g[j]]
    if l1_student.bias is not None:
        if l1_teacher.bias is None:
            l1_student.bias.data.fill_(0)
        else:
            for j in range(student_hidden):
                l1_student.bias.data[j] = l1_teacher.bias.data[g[j]]


def wider_second_(
    l2_teacher: nn.Module,
    l2_student: nn.Module,
    g: List[int]
) -> None:
    student_hidden = l2_student.weight.shape[1]
    teacher_out = l2_teacher.weight.shape[0]
    for j in range(student_hidden):
        l2_student.weight.data[:teacher_out, j] = 1/g.count(g[j]) \
            * l2_teacher.weight.data[:, g[j]]
    if l2_student.bias is not None:
        if l2_teacher.bias is None:
            l2_student.bias.data.fill_(0)
        else:
            for j in range(student_hidden):
                l2_student.bias.data[:teacher_out] = l2_teacher.bias.data


def wider_first(
    l1_teacher: nn.Module,
    hidden_size: int,
    g: List[int],
    # dim: int = 1 # dim to expand
):
    weight = torch.stack([l1_teacher.weight.data[g[j]] for j in range(hidden_size)], dim=0)
    bias = torch.stack([l1_teacher.bias.data[g[j]] for j in range(hidden_size)], dim=0)
    return weight, bias


def wider_second(
    l2_teacher: nn.Module,
    hidden_size: int,
    g: List[int]
):
    weight = torch.stack([1/g.count(g[j]) * l2_teacher.weight.data[:, g[j]] for j in range(hidden_size)], dim=1)
    bias = l2_teacher.bias.data
    return weight, bias