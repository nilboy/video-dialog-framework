import numpy as np

def m_recall(k, rank):
  """
  Args:
    k:  int
    rank: [batch_size] int32
  Returns:
    score: float32
  """
  return np.mean(rank <= k)

def m_mrr(rank):
  return np.mean(1.0/rank)

def m_rank(rank):
  return np.mean(rank)

def get_rank(logits):
  """
  Args:
    logits: [batch_size, candidate_num] np.float32
    rank:   [batch_size] np.int
  """
  rank = (-logits).argsort().argsort() + 1
  rank = np.asarray(rank[:, 0], dtype=np.int32)
  return rank
